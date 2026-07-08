from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Iterable, Iterator, Mapping
from zipfile import ZipFile

import numpy as np
from PIL import Image, UnidentifiedImageError

from backend.app.ml.brain_mri.contract import BrainMRIClass
from backend.app.ml.brain_mri.dataset_split import DatasetSplit


DEFAULT_IMAGE_SIZE = (224, 224)
DEFAULT_CHANNELS = 3

SUPPORTED_IMAGE_SUFFIXES = frozenset(
    {
        ".jpg",
        ".jpeg",
        ".png",
        ".bmp",
        ".tif",
        ".tiff",
    }
)


class ArchiveDatasetError(RuntimeError):
    """Base error for archive-backed dataset failures."""


class ArchiveNotFoundError(ArchiveDatasetError):
    """Raised when a required dataset archive is missing."""


class ArchiveMemberNotFoundError(ArchiveDatasetError):
    """Raised when an assigned archive member is missing."""


class ImageDecodingError(ArchiveDatasetError):
    """Raised when an image payload cannot be decoded safely."""


@dataclass(frozen=True, slots=True)
class ArchiveBackedRecord:
    record_id: str
    source_id: str
    archive_name: str
    archive_member: str
    normalized_class: BrainMRIClass
    content_sha256: str
    split: DatasetSplit


@dataclass(frozen=True, slots=True)
class ImagePreprocessingConfig:
    height: int = DEFAULT_IMAGE_SIZE[0]
    width: int = DEFAULT_IMAGE_SIZE[1]
    channels: int = DEFAULT_CHANNELS

    def __post_init__(self) -> None:
        if self.height <= 0:
            raise ValueError("height must be positive.")

        if self.width <= 0:
            raise ValueError("width must be positive.")

        if self.channels not in (1, 3):
            raise ValueError(
                "channels must be either 1 or 3."
            )


@dataclass(frozen=True, slots=True)
class LoadedImage:
    record: ArchiveBackedRecord
    image: np.ndarray


def build_archive_record_index(
    records: Iterable[ArchiveBackedRecord],
) -> dict[str, ArchiveBackedRecord]:
    index: dict[str, ArchiveBackedRecord] = {}

    for record in records:
        if record.record_id in index:
            raise ValueError(
                f"Duplicate record ID: {record.record_id}"
            )

        index[record.record_id] = record

    return index


def resolve_archive_paths(
    *,
    archive_root: Path,
    archive_names: Iterable[str],
) -> dict[str, Path]:
    if not archive_root.is_dir():
        raise ArchiveNotFoundError(
            f"Archive root does not exist: {archive_root}"
        )

    discovered: dict[str, Path] = {}

    for archive in sorted(archive_root.rglob("*.zip")):
        if archive.name in discovered:
            raise ArchiveDatasetError(
                f"Duplicate archive filename: {archive.name}"
            )

        discovered[archive.name] = archive

    required = tuple(sorted(set(archive_names)))
    missing = tuple(
        name
        for name in required
        if name not in discovered
    )

    if missing:
        raise ArchiveNotFoundError(
            "Missing required archives: "
            + ", ".join(missing)
        )

    return {
        name: discovered[name]
        for name in required
    }


def decode_image_payload(
    payload: bytes,
    *,
    config: ImagePreprocessingConfig = (
        ImagePreprocessingConfig()
    ),
) -> np.ndarray:
    if not payload:
        raise ImageDecodingError("Image payload is empty.")

    try:
        with Image.open(BytesIO(payload)) as image:
            image.load()

            if config.channels == 1:
                image = image.convert("L")
            else:
                image = image.convert("RGB")

            image = image.resize(
                (config.width, config.height),
                resample=Image.Resampling.BILINEAR,
            )

            array = np.asarray(
                image,
                dtype=np.float32,
            )

    except (
        UnidentifiedImageError,
        OSError,
        ValueError,
    ) as exc:
        raise ImageDecodingError(
            "Image payload could not be decoded."
        ) from exc

    if config.channels == 1:
        array = np.expand_dims(array, axis=-1)

    expected_shape = (
        config.height,
        config.width,
        config.channels,
    )

    if array.shape != expected_shape:
        raise ImageDecodingError(
            "Decoded image has unexpected shape: "
            f"{array.shape}; expected {expected_shape}."
        )

    array = array / np.float32(255.0)

    if not np.isfinite(array).all():
        raise ImageDecodingError(
            "Decoded image contains non-finite values."
        )

    return np.ascontiguousarray(
        array,
        dtype=np.float32,
    )


def class_index_mapping() -> dict[BrainMRIClass, int]:
    ordered_classes = (
        BrainMRIClass.GLIOMA,
        BrainMRIClass.MENINGIOMA,
        BrainMRIClass.PITUITARY_TUMOR,
        BrainMRIClass.NO_TUMOR,
    )

    return {
        brain_class: index
        for index, brain_class in enumerate(
            ordered_classes
        )
    }


def class_index(
    brain_class: BrainMRIClass,
) -> int:
    return class_index_mapping()[brain_class]


class ArchiveBackedImageLoader:
    def __init__(
        self,
        *,
        archive_root: Path,
        records: Iterable[ArchiveBackedRecord],
        config: ImagePreprocessingConfig = (
            ImagePreprocessingConfig()
        ),
    ) -> None:
        self._records = tuple(records)
        self._config = config

        self._archive_paths = resolve_archive_paths(
            archive_root=archive_root,
            archive_names=(
                record.archive_name
                for record in self._records
            ),
        )

        self._open_archives: dict[str, ZipFile] = {}

    @property
    def records(self) -> tuple[ArchiveBackedRecord, ...]:
        return self._records

    @property
    def config(self) -> ImagePreprocessingConfig:
        return self._config

    def records_for_split(
        self,
        split: DatasetSplit,
    ) -> tuple[ArchiveBackedRecord, ...]:
        return tuple(
            record
            for record in self._records
            if record.split is split
        )

    def _archive_for(
        self,
        archive_name: str,
    ) -> ZipFile:
        archive = self._open_archives.get(
            archive_name
        )

        if archive is None:
            archive = ZipFile(
                self._archive_paths[archive_name]
            )
            self._open_archives[
                archive_name
            ] = archive

        return archive

    def close(self) -> None:
        for archive in self._open_archives.values():
            archive.close()

        self._open_archives.clear()

    def __enter__(
        self,
    ) -> "ArchiveBackedImageLoader":
        return self

    def __exit__(
        self,
        exc_type: object,
        exc_value: object,
        traceback: object,
    ) -> None:
        self.close()

    def read_payload(
        self,
        record: ArchiveBackedRecord,
    ) -> bytes:
        archive = self._archive_for(
            record.archive_name
        )

        try:
            return archive.read(
                record.archive_member
            )
        except KeyError as exc:
            raise ArchiveMemberNotFoundError(
                "Archive member not found: "
                f"{record.archive_name} :: "
                f"{record.archive_member}"
            ) from exc

    def load_image(
        self,
        record: ArchiveBackedRecord,
    ) -> LoadedImage:
        payload = self.read_payload(record)

        image = decode_image_payload(
            payload,
            config=self._config,
        )

        return LoadedImage(
            record=record,
            image=image,
        )

    def iter_split(
        self,
        split: DatasetSplit,
    ) -> Iterator[LoadedImage]:
        for record in self.records_for_split(split):
            yield self.load_image(record)


def join_canonical_records_with_splits(
    *,
    canonical_records: Iterable[Mapping[str, object]],
    assignments: Iterable[Mapping[str, object]],
) -> tuple[ArchiveBackedRecord, ...]:
    canonical_by_id: dict[str, Mapping[str, object]] = {}

    for canonical in canonical_records:
        record_id = str(canonical["record_id"])

        if record_id in canonical_by_id:
            raise ValueError(
                f"Duplicate canonical record ID: {record_id}"
            )

        canonical_by_id[record_id] = canonical

    assignment_by_id: dict[str, Mapping[str, object]] = {}

    for assignment in assignments:
        record_id = str(assignment["record_id"])

        if record_id in assignment_by_id:
            raise ValueError(
                f"Duplicate split assignment: {record_id}"
            )

        assignment_by_id[record_id] = assignment

    canonical_ids = set(canonical_by_id)
    assignment_ids = set(assignment_by_id)

    if canonical_ids != assignment_ids:
        missing_assignments = sorted(
            canonical_ids - assignment_ids
        )

        unknown_assignments = sorted(
            assignment_ids - canonical_ids
        )

        raise ValueError(
            "Canonical and split record IDs do not match. "
            f"Missing assignments: "
            f"{len(missing_assignments)}; "
            f"unknown assignments: "
            f"{len(unknown_assignments)}."
        )

    records: list[ArchiveBackedRecord] = []

    for record_id in sorted(canonical_ids):
        canonical = canonical_by_id[record_id]
        assignment = assignment_by_id[record_id]

        canonical_hash = str(
            canonical["content_sha256"]
        )

        assignment_hash = str(
            assignment["content_sha256"]
        )

        if canonical_hash != assignment_hash:
            raise ValueError(
                "Content hash mismatch for record: "
                f"{record_id}"
            )

        canonical_class = BrainMRIClass(
            str(canonical["normalized_class"])
        )

        assignment_class = BrainMRIClass(
            str(assignment["normalized_class"])
        )

        if canonical_class is not assignment_class:
            raise ValueError(
                "Class mismatch for record: "
                f"{record_id}"
            )

        records.append(
            ArchiveBackedRecord(
                record_id=record_id,
                source_id=str(
                    canonical["source_id"]
                ),
                archive_name=str(
                    canonical["archive_name"]
                ),
                archive_member=str(
                    canonical["archive_member"]
                ),
                normalized_class=canonical_class,
                content_sha256=canonical_hash,
                split=DatasetSplit(
                    str(assignment["split"])
                ),
            )
        )

    return tuple(records)
