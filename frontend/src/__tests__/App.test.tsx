// @vitest-environment happy-dom

import {
  fireEvent,
  render,
  screen,
  waitFor,
} from '@testing-library/react'
import {
  beforeEach,
  describe,
  expect,
  it,
  vi,
} from 'vitest'

import App from '../App'
import {
  analyzeImage,
  fetchExplanationOverlay,
} from '../api/mediscan'
import type { AnalysisResponse } from '../types/api'

vi.mock('../api/mediscan', () => ({
  analyzeImage: vi.fn(),
  fetchExplanationOverlay: vi.fn(),
}))

const mockedAnalyzeImage =
  vi.mocked(analyzeImage)

const mockedFetchExplanationOverlay =
  vi.mocked(fetchExplanationOverlay)

const analysisResponse: AnalysisResponse = {
  primary_prediction: {
    model: 'baseline_cnn_v1',
    label: 'PNEUMONIA',
    probability: 0.99827528,
    threshold: 0.053277,
  },
  secondary_signal: {
    model: 'advanced_v3',
    role: 'exploratory_safety_signal',
    probability: 0.92649692,
    threshold: 0.5,
    predicted_label: 'PNEUMONIA',
    automatic_override_allowed: false,
  },
  decision: {
    final_label: 'PNEUMONIA',
    source: 'baseline_cnn_v1',
    manual_review_recommended: false,
    warning_code: null,
  },
  preprocessing: {
    v1: 'baseline_v1',
    v3: 'advanced_v3',
    v3_metadata: {},
  },
  explanation: {
    method: 'gradcam',
    mode: 'absolute_attribution',
    last_conv_layer: 'conv2d_3',
    raw_heatmap_shape: [28, 28],
    output_width: 1184,
    output_height: 1128,
    minimum: 0.1,
    maximum: 1,
  },
  explanation_quality: {
    border_energy_ratio: 0.2,
    thorax_energy_ratio: 0.8,
    peak_in_border: 0,
    quality_status: 'ELEVATED_SHORTCUT_RISK',
    display_warning: true,
    warning_code: 'ELEVATED_SHORTCUT_RISK',
    explanation_mode: 'absolute_attribution',
    attribution_note: 'Attribution quality is limited.',
    region_definition: 'heuristic_thorax_region',
  },
  visualization_endpoints: {
    heatmap: '/api/v1/explain',
    overlay: '/api/v1/explain/overlay',
  },
  disclaimer:
    'Educational prototype only. Not for clinical use.',
}

function createFile(
  name = 'xray.png',
): File {
  return new File(
    ['valid-image-content'],
    name,
    {
      type: 'image/png',
    },
  )
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

function selectFile(
  file = createFile(),
) {
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

async function startAnalysis() {
  selectFile()

  fireEvent.click(
    screen.getByRole(
      'button',
      {
        name: 'Analyze X-ray',
      },
    ),
  )
}

describe('App analysis orchestration', () => {
  beforeEach(() => {
    vi.stubGlobal(
      'URL',
      {
        createObjectURL: vi
          .fn()
          .mockReturnValueOnce('blob:preview'),
        revokeObjectURL: vi.fn(),
      },
    )
  })

  it(
    'renders a successful analysis and overlay',
    async () => {
      mockedAnalyzeImage.mockResolvedValue(
        analysisResponse,
      )

      mockedFetchExplanationOverlay
        .mockResolvedValue('blob:overlay')

      render(<App />)

      await startAnalysis()

      expect(
        await screen.findByText(
          'Authoritative result',
        ),
      ).toBeInTheDocument()

      expect(
        screen.getByText('99.8%'),
      ).toBeInTheDocument()

      expect(
        screen.getByAltText(
          'Grad-CAM explanation overlay',
        ),
      ).toHaveAttribute(
        'src',
        'blob:overlay',
      )

      expect(
        mockedAnalyzeImage,
      ).toHaveBeenCalledTimes(1)

      expect(
        mockedFetchExplanationOverlay,
      ).toHaveBeenCalledTimes(1)
    },
  )

  it(
    'shows a fatal analysis error',
    async () => {
      mockedAnalyzeImage.mockRejectedValue(
        new Error(
          'Combined analysis could not be completed.',
        ),
      )

      render(<App />)

      await startAnalysis()

      expect(
        await screen.findByRole('alert'),
      ).toHaveTextContent(
        'Combined analysis could not be completed.',
      )

      expect(
        mockedFetchExplanationOverlay,
      ).not.toHaveBeenCalled()
    },
  )

  it(
    'preserves analysis when overlay loading fails',
    async () => {
      mockedAnalyzeImage.mockResolvedValue(
        analysisResponse,
      )

      mockedFetchExplanationOverlay
        .mockRejectedValue(
          new Error('Overlay unavailable'),
        )

      render(<App />)

      await startAnalysis()

      expect(
        await screen.findByText(
          'Authoritative result',
        ),
      ).toBeInTheDocument()

      expect(
        await screen.findByRole('status'),
      ).toHaveTextContent(
        'Analysis completed successfully, but the Grad-CAM overlay could not be loaded.',
      )

      expect(
        screen.getByText(
          'Explanation visualization unavailable.',
        ),
      ).toBeInTheDocument()
    },
  )

  it(
    'clears the completed analysis',
    async () => {
      mockedAnalyzeImage.mockResolvedValue(
        analysisResponse,
      )

      mockedFetchExplanationOverlay
        .mockResolvedValue('blob:overlay')

      render(<App />)

      await startAnalysis()

      const clearButton =
        await screen.findByRole(
          'button',
          {
            name: 'Analyze another X-ray',
          },
        )

      fireEvent.click(clearButton)

      expect(
        screen.queryByText(
          'Authoritative result',
        ),
      ).not.toBeInTheDocument()

      expect(
        screen.getByRole(
          'button',
          {
            name: 'Select X-ray',
          },
        ),
      ).toBeInTheDocument()
    },
  )

  it(
    'discards a stale analysis response after clear',
    async () => {
      let resolveAnalysis:
        | ((
            value: AnalysisResponse,
          ) => void)
        | undefined

      mockedAnalyzeImage.mockImplementation(
        () =>
          new Promise<AnalysisResponse>(
            (resolve) => {
              resolveAnalysis = resolve
            },
          ),
      )

      render(<App />)

      await startAnalysis()

      fireEvent.click(
        screen.getByRole(
          'button',
          {
            name: 'Remove selected image',
          },
        ),
      )

      resolveAnalysis?.(analysisResponse)

      await waitFor(() => {
        expect(
          screen.queryByText(
            'Authoritative result',
          ),
        ).not.toBeInTheDocument()
      })

      expect(
        mockedFetchExplanationOverlay,
      ).not.toHaveBeenCalled()
    },
  )
})
