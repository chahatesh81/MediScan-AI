// @vitest-environment happy-dom

import {
  fireEvent,
  render,
  screen,
} from '@testing-library/react'
import {
  describe,
  expect,
  it,
  vi,
} from 'vitest'

import UploadZone from '../UploadZone'

interface RenderOptions {
  file?: File | null
  previewUrl?: string | null
  disabled?: boolean
  onClear?: () => void
}

function renderUploadZone({
  file = null,
  previewUrl = null,
  disabled = false,
  onClear = vi.fn(),
}: RenderOptions = {}) {
  const onFileSelect = vi.fn()
  const onValidationError = vi.fn()

  render(
    <UploadZone
      file={file}
      previewUrl={previewUrl}
      disabled={disabled}
      onFileSelect={onFileSelect}
      onClear={onClear}
      onValidationError={onValidationError}
    />,
  )

  return {
    onFileSelect,
    onClear,
    onValidationError,
  }
}

function getFileInput(): HTMLInputElement {
  const input = document.querySelector(
    'input[type="file"]',
  )

  if (!(input instanceof HTMLInputElement)) {
    throw new Error(
      'Upload file input was not found.',
    )
  }

  return input
}

function selectFile(file: File) {
  const input = getFileInput()

  Object.defineProperty(
    input,
    'files',
    {
      configurable: true,
      value: [file],
    },
  )

  fireEvent.change(input)
}

describe('UploadZone', () => {
  it('accepts a valid JPEG image', () => {
    const {
      onFileSelect,
      onValidationError,
    } = renderUploadZone()

    const file = new File(
      ['valid-jpeg-content'],
      'chest-xray.jpg',
      {
        type: 'image/jpeg',
      },
    )

    selectFile(file)

    expect(onFileSelect).toHaveBeenCalledOnce()
    expect(onFileSelect).toHaveBeenCalledWith(file)
    expect(
      onValidationError,
    ).not.toHaveBeenCalled()
  })

  it('accepts a valid PNG image', () => {
    const {
      onFileSelect,
      onValidationError,
    } = renderUploadZone()

    const file = new File(
      ['valid-png-content'],
      'chest-xray.png',
      {
        type: 'image/png',
      },
    )

    selectFile(file)

    expect(onFileSelect).toHaveBeenCalledWith(file)
    expect(
      onValidationError,
    ).not.toHaveBeenCalled()
  })

  it('rejects an unsupported image type', () => {
    const {
      onFileSelect,
      onValidationError,
    } = renderUploadZone()

    const file = new File(
      ['unsupported-content'],
      'chest-xray.webp',
      {
        type: 'image/webp',
      },
    )

    selectFile(file)

    expect(onFileSelect).not.toHaveBeenCalled()
    expect(
      onValidationError,
    ).toHaveBeenCalledWith(
      'Unsupported image type. Use a JPEG or PNG image.',
    )
  })

  it('rejects an empty image', () => {
    const {
      onFileSelect,
      onValidationError,
    } = renderUploadZone()

    const file = new File(
      [],
      'empty.png',
      {
        type: 'image/png',
      },
    )

    selectFile(file)

    expect(onFileSelect).not.toHaveBeenCalled()
    expect(
      onValidationError,
    ).toHaveBeenCalledWith(
      'The selected image is empty.',
    )
  })

  it('rejects an image larger than 10 MB', () => {
    const {
      onFileSelect,
      onValidationError,
    } = renderUploadZone()

    const oversizedContent =
      new Uint8Array(
        10 * 1024 * 1024 + 1,
      )

    const file = new File(
      [oversizedContent],
      'oversized.png',
      {
        type: 'image/png',
      },
    )

    selectFile(file)

    expect(onFileSelect).not.toHaveBeenCalled()
    expect(
      onValidationError,
    ).toHaveBeenCalledWith(
      'Image exceeds the 10 MB upload limit.',
    )
  })

  it('accepts a valid dropped image', () => {
    const {
      onFileSelect,
      onValidationError,
    } = renderUploadZone()

    const file = new File(
      ['dropped-image'],
      'dropped-xray.jpg',
      {
        type: 'image/jpeg',
      },
    )

    const dropZone = screen
      .getByText('Upload a chest X-ray')
      .closest('div.rounded-3xl')

    if (!(dropZone instanceof HTMLDivElement)) {
      throw new Error(
        'Upload drop zone was not found.',
      )
    }

    fireEvent.drop(
      dropZone,
      {
        dataTransfer: {
          files: [file],
        },
      },
    )

    expect(onFileSelect).toHaveBeenCalledWith(file)
    expect(
      onValidationError,
    ).not.toHaveBeenCalled()
  })

  it('ignores dropped files while disabled', () => {
    const {
      onFileSelect,
      onValidationError,
    } = renderUploadZone({
      disabled: true,
    })

    const file = new File(
      ['dropped-image'],
      'dropped-xray.jpg',
      {
        type: 'image/jpeg',
      },
    )

    const dropZone = screen
      .getByText('Upload a chest X-ray')
      .closest('div.rounded-3xl')

    if (!(dropZone instanceof HTMLDivElement)) {
      throw new Error(
        'Upload drop zone was not found.',
      )
    }

    fireEvent.drop(
      dropZone,
      {
        dataTransfer: {
          files: [file],
        },
      },
    )

    expect(onFileSelect).not.toHaveBeenCalled()
    expect(
      onValidationError,
    ).not.toHaveBeenCalled()
  })

  it('renders the selected image preview', () => {
    const file = new File(
      ['preview-image'],
      'patient-xray.png',
      {
        type: 'image/png',
      },
    )

    renderUploadZone({
      file,
      previewUrl: 'blob:preview-image',
    })

    expect(
      screen.getByText('patient-xray.png'),
    ).toBeInTheDocument()

    const preview = screen.getByAltText(
      'Selected chest X-ray preview',
    )

    expect(preview).toHaveAttribute(
      'src',
      'blob:preview-image',
    )
  })

  it('clears the selected image', () => {
    const file = new File(
      ['preview-image'],
      'patient-xray.png',
      {
        type: 'image/png',
      },
    )

    const {
      onClear,
    } = renderUploadZone({
      file,
      previewUrl: 'blob:preview-image',
    })

    fireEvent.click(
      screen.getByRole(
        'button',
        {
          name: 'Remove selected image',
        },
      ),
    )

    expect(onClear).toHaveBeenCalledOnce()
  })

  it('allows clearing the image while disabled', () => {
    const onClear = vi.fn()

    const file = new File(
      ['preview-image'],
      'patient-xray.png',
      {
        type: 'image/png',
      },
    )

    renderUploadZone({
      file,
      previewUrl: 'blob:preview-image',
      disabled: true,
      onClear,
    })

    const clearButton = screen.getByRole(
      'button',
      {
        name: 'Remove selected image',
      },
    )

    expect(clearButton).toBeEnabled()

    fireEvent.click(clearButton)

    expect(onClear).toHaveBeenCalledOnce()
  })
})
