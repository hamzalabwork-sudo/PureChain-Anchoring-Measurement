// SPDX-License-Identifier: MIT
pragma solidity 0.8.19;

/// @title FinVerityRegistry
/// @notice Permissioned registry of cryptographic commitments to off-chain financial audit evidence.
///         Only the SHA-256 commitment H_i and minimal metadata are stored; the evidence package
///         E_i itself stays in off-chain storage. Every record ID can be committed at most once,
///         whether on its own (recordEvidence) or inside a Merkle batch (recordBatch).
contract FinVerityRegistry {
    struct Record {
        bytes32 evidenceHash; // H_i = SHA256(Canon(E_i))
        bytes32 metaHash;     // commitment to authorization / validation metadata (A_i, V_i)
        address submitter;    // account that submitted the record
        address approver;     // account whose signature S_i authorized the record
        uint64 timestamp;     // t_i, block timestamp at inclusion
        uint64 blockNumber;
        uint8 level;          // evidence complexity level (1-4)
    }

    address public immutable admin;
    mapping(address => bool) public isSubmitter;
    mapping(address => bool) public isApprover;
    mapping(bytes32 => Record) private records;
    /// record ID -> the batch that committed it (records committed on their own are in `records`)
    mapping(bytes32 => bytes32) public committedIn;
    uint256 public recordCount;

    // secp256k1n / 2, used to reject malleable signatures
    uint256 private constant HALF_N = 0x7FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF5D576E7357A4501DDFE92F46681B20A0;
    uint8 private constant KIND_RECORD = 0;
    uint8 private constant KIND_BATCH = 1;

    event EvidenceRecorded(
        bytes32 indexed recordId,
        bytes32 evidenceHash,
        address indexed submitter,
        address indexed approver,
        uint64 timestamp
    );
    event RolesUpdated(address indexed account, bool submitter, bool approver);

    error NotAdmin();
    error NotSubmitter();
    error AlreadyRecorded(bytes32 recordId);
    error InvalidApproval();
    error EmptyBatch();

    constructor() {
        admin = msg.sender;
        isSubmitter[msg.sender] = true;
    }

    function setRoles(address[] calldata accounts, bool submitter, bool approver) external {
        if (msg.sender != admin) revert NotAdmin();
        for (uint256 i = 0; i < accounts.length; i++) {
            isSubmitter[accounts[i]] = submitter;
            isApprover[accounts[i]] = approver;
            emit RolesUpdated(accounts[i], submitter, approver);
        }
    }

    /// @notice Digest the approver signs: an EIP-191 personal_sign over the abi.encode'd (fixed-width, unambiguous)
    ///         tuple (chain, registry, submitter, kind, id, hash, metaHash, level, count). Binding the submitter
    ///         stops another registered submitter from replaying a pending approval; binding count fixes the
    ///         number of leaves under a batch root.
    function approvalDigest(address submitter, uint8 kind, bytes32 id, bytes32 hash, bytes32 metaHash, uint8 level,
                            uint32 count) public view returns (bytes32) {
        bytes32 inner = keccak256(abi.encode(block.chainid, address(this), submitter, kind, id, hash, metaHash, level,
                                             count));
        return keccak256(abi.encodePacked("\x19Ethereum Signed Message:\n32", inner));
    }

    function _unused(bytes32 recordId) private view returns (bool) {
        return records[recordId].timestamp == 0 && committedIn[recordId] == bytes32(0);
    }

    function recordEvidence(
        bytes32 recordId,
        bytes32 evidenceHash,
        bytes32 metaHash,
        uint8 level,
        bytes calldata approval
    ) external {
        if (!isSubmitter[msg.sender]) revert NotSubmitter();
        if (!_unused(recordId)) revert AlreadyRecorded(recordId);
        address approver = _recover(approvalDigest(msg.sender, KIND_RECORD, recordId, evidenceHash, metaHash, level, 1),
                                    approval);
        if (approver == address(0) || !isApprover[approver]) revert InvalidApproval();

        records[recordId] = Record({
            evidenceHash: evidenceHash,
            metaHash: metaHash,
            submitter: msg.sender,
            approver: approver,
            timestamp: uint64(block.timestamp),
            blockNumber: uint64(block.number),
            level: level
        });
        unchecked { recordCount++; }
        emit EvidenceRecorded(recordId, evidenceHash, msg.sender, approver, uint64(block.timestamp));
    }

    // ------------------------------------------------------------------ batched commitments
    struct Batch {
        bytes32 merkleRoot;   // root over leaves keccak256(0x00 || recordId || evidenceHash || metaHash)
        bytes32 metaHash;
        address submitter;
        address approver;
        uint64 timestamp;
        uint64 blockNumber;
        uint32 count;
        uint8 level;
    }

    mapping(bytes32 => Batch) private batches;
    uint256 public batchCount;

    event BatchRecorded(bytes32 indexed batchId, bytes32 merkleRoot, uint32 count, address indexed submitter,
                        address indexed approver, uint64 timestamp);

    /// @notice Commit the records `recordIds` with one transaction (one Merkle root). Every ID is marked as committed
    ///         in this batch and may never be committed again, so an insider cannot commit two versions of a record
    ///         in different batches. The approver signs approvalDigest(submitter, 1, batchId, merkleRoot, metaHash,
    ///         level, recordIds.length).
    function recordBatch(
        bytes32 batchId,
        bytes32 merkleRoot,
        bytes32[] calldata recordIds,
        bytes32 metaHash,
        uint8 level,
        bytes calldata approval
    ) external {
        uint32 count = uint32(recordIds.length);
        for (uint256 i = 0; i < count; i++) {
            bytes32 rid = recordIds[i];
            if (!_unused(rid)) revert AlreadyRecorded(rid);
            committedIn[rid] = batchId;
        }
        _storeBatch(batchId, merkleRoot, count, metaHash, level, approval);
    }

    event BatchMembers(bytes32 indexed batchId, bytes32[] recordIds);

    /// @notice Cheaper alternative to recordBatch: the record IDs are only logged (BatchMembers), not stored, so the
    ///         contract cannot reject a re-committed ID. An auditor who reads every BatchMembers log of the registry
    ///         (which the insider can neither alter nor withhold) detects an ID committed twice instead.
    function recordBatchLogged(
        bytes32 batchId,
        bytes32 merkleRoot,
        bytes32[] calldata recordIds,
        bytes32 metaHash,
        uint8 level,
        bytes calldata approval
    ) external {
        _storeBatch(batchId, merkleRoot, uint32(recordIds.length), metaHash, level, approval);
        emit BatchMembers(batchId, recordIds);
    }

    function _storeBatch(bytes32 batchId, bytes32 merkleRoot, uint32 count, bytes32 metaHash, uint8 level,
                         bytes calldata approval) private {
        if (!isSubmitter[msg.sender]) revert NotSubmitter();
        if (count == 0) revert EmptyBatch();
        if (batches[batchId].timestamp != 0) revert AlreadyRecorded(batchId);
        address approver = _recover(approvalDigest(msg.sender, KIND_BATCH, batchId, merkleRoot, metaHash, level, count),
                                    approval);
        if (approver == address(0) || !isApprover[approver]) revert InvalidApproval();
        batches[batchId] = Batch({
            merkleRoot: merkleRoot, metaHash: metaHash, submitter: msg.sender, approver: approver,
            timestamp: uint64(block.timestamp), blockNumber: uint64(block.number), count: count, level: level
        });
        unchecked { batchCount++; recordCount += count; }
        emit BatchRecorded(batchId, merkleRoot, count, msg.sender, approver, uint64(block.timestamp));
    }

    function getBatch(bytes32 batchId) external view returns (Batch memory) {
        return batches[batchId];
    }

    /// @notice On-chain inclusion check of one record in a committed batch: the record ID must have been committed in
    ///         this batch, and its leaf must hash to the batch root (leaves and nodes are domain-separated).
    function verifyInclusion(bytes32 batchId, bytes32 recordId, bytes32 evidenceHash, bytes32 metaHash,
                             bytes32[] calldata proof) external view returns (bool) {
        Batch storage b = batches[batchId];
        if (b.timestamp == 0 || committedIn[recordId] != batchId) return false;
        bytes32 h = keccak256(abi.encodePacked(bytes1(0x00), recordId, evidenceHash, metaHash));
        for (uint256 i = 0; i < proof.length; i++) {
            bytes32 p = proof[i];
            h = h < p ? keccak256(abi.encodePacked(bytes1(0x01), h, p)) : keccak256(abi.encodePacked(bytes1(0x01), p, h));
        }
        return h == b.merkleRoot;
    }

    function getRecord(bytes32 recordId) external view returns (Record memory) {
        return records[recordId];
    }

    /// @notice Verify(E_i') = 1 iff H(E_i') equals the recorded commitment.
    function verifyEvidence(bytes32 recordId, bytes32 candidateHash) external view returns (bool) {
        Record storage r = records[recordId];
        return r.timestamp != 0 && r.evidenceHash == candidateHash;
    }

    function _recover(bytes32 digest, bytes calldata sig) private pure returns (address) {
        if (sig.length != 65) return address(0);
        bytes32 r = bytes32(sig[0:32]);
        bytes32 s = bytes32(sig[32:64]);
        uint8 v = uint8(sig[64]);
        if (v < 27) v += 27;
        if (uint256(s) > HALF_N || (v != 27 && v != 28)) return address(0);
        return ecrecover(digest, v, r, s);
    }
}
