import {
  FileImage,
  UploadCloud,
  X,
} from 'lucide-react'
import {
  useRef,
  type ChangeEvent,
  type DragEvent,
} from 'react'

interface UploadZoneProps {
  file: File | null
  previewUrl: string | null
  disabled: boolean
  onFileSelect: (file: File) => void
  onClear: () => void
  onValidationError: (message: string) => void
}

const ACCEPTED_TYPES = [
  'image/jpeg',
  'image/png',
]

const MAX_UPLOAD_BYTES = 10 * 1024 * 1024

function UploadZone({
  file,
  previewUrl,
  disabled,
  onFileSelect,
  onClear,
  onValidationError,
}: UploadZoneProps) {
  const inputRef = useRef<HTMLInputElement>(null)

  function handleFile(fileCandidate?: File) {
    if (!fileCandidate) {
      return
    }

    if (!ACCEPTED_TYPES.includes(fileCandidate.type)) {
      onValidationError(
        'Unsupported image type. Use a JPEG or PNG image.',
      )
      return
    }

    if (fileCandidate.size === 0) {
      onValidationError(
        'The selected image is empty.',
      )
      return
    }

    if (fileCandidate.size > MAX_UPLOAD_BYTES) {
      onValidationError(
        'Image exceeds the 10 MB upload limit.',
      )
      return
    }

    onFileSelect(fileCandidate)
  }

  function handleInputChange(
    event: ChangeEvent<HTMLInputElement>,
  ) {
    handleFile(event.target.files?.[0])
    event.target.value = ''
  }

  function handleDrop(
    event: DragEvent<HTMLDivElement>,
  ) {
    event.preventDefault()

    if (disabled) {
      return
    }

    handleFile(event.dataTransfer.files?.[0])
  }

  if (file && previewUrl) {
    return (
      <div className="overflow-hidden rounded-3xl border border-slate-800 bg-slate-900/70">
        <div className="flex items-center justify-between border-b border-slate-800 px-5 py-4">
          <div className="flex min-w-0 items-center gap-3">
            <FileImage className="size-5 shrink-0 text-cyan-400" />

            <div className="min-w-0">
              <p className="truncate text-sm font-medium text-slate-200">
                {file.name}
              </p>

              <p className="text-xs text-slate-500">
                {(file.size / 1024 / 1024).toFixed(2)} MB
              </p>
            </div>
          </div>

          <button
            type="button"
            onClick={onClear}
            className="rounded-xl border border-slate-700 p-2 text-slate-400 transition hover:border-slate-600 hover:text-white"
            aria-label="Remove selected image"
          >
            <X className="size-4" />
          </button>
        </div>

        <div className="flex min-h-80 items-center justify-center bg-black/30 p-4">
          <img
            src={previewUrl}
            alt="Selected chest X-ray preview"
            className="max-h-[520px] w-full rounded-2xl object-contain"
          />
        </div>
      </div>
    )
  }

  return (
    <div
      onDragOver={(event) => event.preventDefault()}
      onDrop={handleDrop}
      className="rounded-3xl border border-dashed border-slate-700 bg-slate-900/40 p-8 transition hover:border-cyan-500/50 hover:bg-slate-900/70"
    >
      <div className="flex min-h-72 flex-col items-center justify-center text-center">
        <div className="flex size-16 items-center justify-center rounded-2xl bg-cyan-500/10">
          <UploadCloud className="size-8 text-cyan-400" />
        </div>

        <h2 className="mt-6 text-xl font-semibold">
          Upload a chest X-ray
        </h2>

        <p className="mt-2 max-w-md text-sm leading-6 text-slate-400">
          Drag and drop an image here, or select a file
          from your device.
        </p>

        <button
          type="button"
          onClick={() => inputRef.current?.click()}
          className="mt-6 rounded-xl bg-cyan-400 px-5 py-3 text-sm font-semibold text-slate-950 transition hover:bg-cyan-300 disabled:cursor-not-allowed disabled:opacity-50"
        >
          Select X-ray
        </button>

        <p className="mt-4 text-xs text-slate-500">
          JPEG and PNG images · Maximum 10 MB
        </p>

        <input
          ref={inputRef}
          type="file"
          accept="image/jpeg,image/png"
          onChange={handleInputChange}
          className="hidden"
        />
      </div>
    </div>
  )
}

export default UploadZone
