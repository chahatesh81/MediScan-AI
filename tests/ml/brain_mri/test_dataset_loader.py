from __future__ import annotations

from io import BytesIO
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pytest
from PIL import Image

from backend.app.ml.brain_mri.contract import BrainMRIClass
from backend.app.ml.brain_mri.dataset_loader import (
    ArchiveBackedImageLoader,
    ArchiveBackedRecord,
    ArchiveMemberNotFoundError,
    ArchiveNotFoundError,
    ImageDecodingError,
    ImagePreprocessingConfig,
    build_archive_record_index,
    class_index,
    class_index_mapping,
    decode_image_payload,
    join_canonical_records_with_splits,
    resolve_archive_paths,
)
from backend.app.ml.brain_mri.dataset_split import (
    DatasetSplit,
)


def make_image_payload(
    *,
    mode: str = "RGB",
    size: tuple[int, int] = (16, 12),
) -> bytes:
    if mode == "L":
        image = Image.new(mode, size, color=128)
    else:
        image = Image.new(
            mode,
            size,
            color=(10, 20, 30),
        )

    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def make_record(
    *,
    record_id: str = "record-1",
    archive_name: str = "dataset.zip",
    archive_member: str = "class/image.png",
    split: DatasetSplit = DatasetSplit.TRAIN,
) -> ArchiveBackedRecord:
    return ArchiveBackedRecord(
        record_id=record_id,
        source_id="source",
        archive_name=archive_name,
        archive_member=archive_member,
        normalized_class=BrainMRIClass.GLIOMA,
        content_sha256="a" * 64,
        split=split,
    )


def test_preprocessing_config_defaults() -> None:
    config = ImagePreprocessingConfig()

    assert config.height == 224
    assert config.width == 224
    assert config.channels == 3


@pytest.mark.parametrize(
    "kwargs",
    (
        {"height": 0},
        {"height": -1},
        {"width": 0},
        {"width": -1},
        {"channels": 2},
        {"channels": 4},
    ),
)
def test_preprocessing_config_rejects_invalid_values(
    kwargs: dict[str, int],
) -> None:
    with pytest.raises(ValueError):
        ImagePreprocessingConfig(**kwargs)


def test_decode_rgb_image() -> None:
    result = decode_image_payload(
        make_image_payload(),
        config=ImagePreprocessingConfig(
            height=32,
            width=24,
            channels=3,
        ),
    )

    assert result.shape == (32, 24, 3)
    assert result.dtype == np.float32
    assert result.flags.c_contiguous
    assert float(result.min()) >= 0.0
    assert float(result.max()) <= 1.0


def test_decode_grayscale_to_rgb() -> None:
    result = decode_image_payload(
        make_image_payload(mode="L"),
        config=ImagePreprocessingConfig(
            height=20,
            width=18,
            channels=3,
        ),
    )

    assert result.shape == (20, 18, 3)


def test_decode_rgb_to_grayscale() -> None:
    result = decode_image_payload(
        make_image_payload(),
        config=ImagePreprocessingConfig(
            height=20,
            width=18,
            channels=1,
        ),
    )

    assert result.shape == (20, 18, 1)


def test_empty_payload_is_rejected() -> None:
    with pytest.raises(
        ImageDecodingError,
        match="empty",
    ):
        decode_image_payload(b"")


def test_invalid_payload_is_rejected() -> None:
    with pytest.raises(ImageDecodingError):
        decode_image_payload(b"not an image")


def test_class_index_mapping_is_stable() -> None:
    assert class_index_mapping() == {
        BrainMRIClass.GLIOMA: 0,
        BrainMRIClass.MENINGIOMA: 1,
        BrainMRIClass.PITUITARY_TUMOR: 2,
        BrainMRIClass.NO_TUMOR: 3,
    }


@pytest.mark.parametrize(
    ("brain_class", "expected"),
    (
        (BrainMRIClass.GLIOMA, 0),
        (BrainMRIClass.MENINGIOMA, 1),
        (BrainMRIClass.PITUITARY_TUMOR, 2),
        (BrainMRIClass.NO_TUMOR, 3),
    ),
)
def test_class_index(
    brain_class: BrainMRIClass,
    expected: int,
) -> None:
    assert class_index(brain_class) == expected


def test_record_index_rejects_duplicate_ids() -> None:
    record = make_record()

    with pytest.raises(
        ValueError,
        match="Duplicate record ID",
    ):
        build_archive_record_index(
            (record, record)
        )


def test_resolve_archive_paths(
    tmp_path: Path,
) -> None:
    archive = tmp_path / "nested" / "dataset.zip"
    archive.parent.mkdir()
    archive.write_bytes(b"")

    result = resolve_archive_paths(
        archive_root=tmp_path,
        archive_names=("dataset.zip",),
    )

    assert result == {
        "dataset.zip": archive
    }


def test_missing_archive_is_rejected(
    tmp_path: Path,
) -> None:
    with pytest.raises(ArchiveNotFoundError):
        resolve_archive_paths(
            archive_root=tmp_path,
            archive_names=("missing.zip",),
        )


def test_duplicate_archive_names_are_rejected(
    tmp_path: Path,
) -> None:
    first = tmp_path / "a" / "dataset.zip"
    second = tmp_path / "b" / "dataset.zip"

    first.parent.mkdir()
    second.parent.mkdir()

    first.write_bytes(b"")
    second.write_bytes(b"")

    with pytest.raises(
        Exception,
        match="Duplicate archive filename",
    ):
        resolve_archive_paths(
            archive_root=tmp_path,
            archive_names=("dataset.zip",),
        )


def test_loader_reads_and_decodes_image(
    tmp_path: Path,
) -> None:
    archive = tmp_path / "dataset.zip"
    payload = make_image_payload()

    with ZipFile(archive, "w") as zip_file:
        zip_file.writestr(
            "class/image.png",
            payload,
        )

    record = make_record()

    loader = ArchiveBackedImageLoader(
        archive_root=tmp_path,
        records=(record,),
        config=ImagePreprocessingConfig(
            height=28,
            width=30,
            channels=3,
        ),
    )

    loaded = loader.load_image(record)

    assert loaded.record is record
    assert loaded.image.shape == (28, 30, 3)
    assert loaded.image.dtype == np.float32


def test_loader_filters_by_split(
    tmp_path: Path,
) -> None:
    archive = tmp_path / "dataset.zip"

    with ZipFile(archive, "w") as zip_file:
        zip_file.writestr(
            "class/image.png",
            make_image_payload(),
        )

    train = make_record(
        record_id="train",
        split=DatasetSplit.TRAIN,
    )

    test = make_record(
        record_id="test",
        split=DatasetSplit.TEST,
    )

    loader = ArchiveBackedImageLoader(
        archive_root=tmp_path,
        records=(test, train),
    )

    assert loader.records_for_split(
        DatasetSplit.TRAIN
    ) == (train,)

    assert loader.records_for_split(
        DatasetSplit.TEST
    ) == (test,)


def test_missing_archive_member_is_rejected(
    tmp_path: Path,
) -> None:
    archive = tmp_path / "dataset.zip"

    with ZipFile(archive, "w"):
        pass

    record = make_record()

    loader = ArchiveBackedImageLoader(
        archive_root=tmp_path,
        records=(record,),
    )

    with pytest.raises(
        ArchiveMemberNotFoundError
    ):
        loader.load_image(record)


def canonical_record() -> dict[str, object]:
    return {
        "record_id": "record-1",
        "source_id": "source",
        "archive_name": "dataset.zip",
        "archive_member": "class/image.jpg",
        "normalized_class": "GLIOMA",
        "content_sha256": "a" * 64,
    }


def split_assignment() -> dict[str, object]:
    return {
        "record_id": "record-1",
        "content_sha256": "a" * 64,
        "normalized_class": "GLIOMA",
        "split": "TRAIN",
    }


def test_join_canonical_records_with_splits() -> None:
    result = join_canonical_records_with_splits(
        canonical_records=(canonical_record(),),
        assignments=(split_assignment(),),
    )

    assert len(result) == 1

    record = result[0]

    assert record.record_id == "record-1"
    assert record.source_id == "source"
    assert record.archive_name == "dataset.zip"
    assert record.archive_member == "class/image.jpg"
    assert record.normalized_class is BrainMRIClass.GLIOMA
    assert record.split is DatasetSplit.TRAIN


def test_join_is_input_order_stable() -> None:
    canonical_a = canonical_record()
    canonical_b = {
        **canonical_record(),
        "record_id": "record-2",
        "content_sha256": "b" * 64,
    }

    assignment_a = split_assignment()
    assignment_b = {
        **split_assignment(),
        "record_id": "record-2",
        "content_sha256": "b" * 64,
        "split": "TEST",
    }

    forward = join_canonical_records_with_splits(
        canonical_records=(
            canonical_a,
            canonical_b,
        ),
        assignments=(
            assignment_a,
            assignment_b,
        ),
    )

    reversed_result = (
        join_canonical_records_with_splits(
            canonical_records=(
                canonical_b,
                canonical_a,
            ),
            assignments=(
                assignment_b,
                assignment_a,
            ),
        )
    )

    assert forward == reversed_result


def test_join_rejects_record_id_mismatch() -> None:
    assignment = {
        **split_assignment(),
        "record_id": "unknown",
    }

    with pytest.raises(
        ValueError,
        match="do not match",
    ):
        join_canonical_records_with_splits(
            canonical_records=(canonical_record(),),
            assignments=(assignment,),
        )


def test_join_rejects_hash_mismatch() -> None:
    assignment = {
        **split_assignment(),
        "content_sha256": "b" * 64,
    }

    with pytest.raises(
        ValueError,
        match="Content hash mismatch",
    ):
        join_canonical_records_with_splits(
            canonical_records=(canonical_record(),),
            assignments=(assignment,),
        )


def test_join_rejects_class_mismatch() -> None:
    assignment = {
        **split_assignment(),
        "normalized_class": "MENINGIOMA",
    }

    with pytest.raises(
        ValueError,
        match="Class mismatch",
    ):
        join_canonical_records_with_splits(
            canonical_records=(canonical_record(),),
            assignments=(assignment,),
        )


def test_join_rejects_duplicate_canonical_ids() -> None:
    canonical = canonical_record()

    with pytest.raises(
        ValueError,
        match="Duplicate canonical record ID",
    ):
        join_canonical_records_with_splits(
            canonical_records=(
                canonical,
                canonical,
            ),
            assignments=(split_assignment(),),
        )


def test_join_rejects_duplicate_assignments() -> None:
    assignment = split_assignment()

    with pytest.raises(
        ValueError,
        match="Duplicate split assignment",
    ):
        join_canonical_records_with_splits(
            canonical_records=(canonical_record(),),
            assignments=(
                assignment,
                assignment,
            ),
        )
