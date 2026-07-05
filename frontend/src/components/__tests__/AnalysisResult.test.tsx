// @vitest-environment happy-dom

import {
  render,
  screen,
} from '@testing-library/react'
import {
  describe,
  expect,
  it,
} from 'vitest'

import type { AnalysisResponse } from '../../types/api'
import AnalysisResult from '../AnalysisResult'

const baseResult: AnalysisResponse = {
  primary_prediction: {
    model: 'baseline_cnn_v1',
    label: 'PNEUMONIA',
    probability: 0.91,
    threshold: 0.053277,
  },
  secondary_signal: {
    model: 'advanced_v3',
    role: 'exploratory_safety_signal',
    probability: 0.84,
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
    mode: 'positive_gradcam',
    last_conv_layer: 'conv2d_3',
    raw_heatmap_shape: [28, 28],
    output_width: 1184,
    output_height: 1128,
    minimum: 0,
    maximum: 1,
  },
  explanation_quality: {
    border_energy_ratio: 0.2,
    thorax_energy_ratio: 0.8,
    peak_in_border: 0,
    quality_status: 'LIMITED_SPATIAL_RELIABILITY',
    display_warning: false,
    warning_code: null,
    explanation_mode: 'positive_gradcam',
    attribution_note: null,
    region_definition: 'heuristic_thorax_region',
  },
  visualization_endpoints: {
    heatmap: '/api/v1/explain',
    overlay: '/api/v1/explain/overlay',
  },
  disclaimer:
    'Educational prototype only. Not for clinical use.',
}

function renderResult(
  overrides: Partial<AnalysisResponse> = {},
) {
  const result: AnalysisResponse = {
    ...baseResult,
    ...overrides,
  }

  return render(
    <AnalysisResult
      result={result}
      originalUrl="blob:original"
      overlayUrl="blob:overlay"
    />,
  )
}

describe('AnalysisResult safety states', () => {
  it('renders the authoritative V1 decision', () => {
    renderResult()

    expect(
      screen.getByText('Authoritative result'),
    ).toBeInTheDocument()

    expect(
      screen.getAllByText('PNEUMONIA').length,
    ).toBeGreaterThanOrEqual(1)

    expect(
      screen.getByText('baseline_cnn_v1'),
    ).toBeInTheDocument()
  })

  it('shows model agreement without manual review', () => {
    renderResult()

    expect(
      screen.getByText('Models agree'),
    ).toBeInTheDocument()

    expect(
      screen.queryByText(
        'Manual review recommended',
      ),
    ).not.toBeInTheDocument()
  })

  it('shows disagreement and manual review when recommended', () => {
    renderResult({
      secondary_signal: {
        ...baseResult.secondary_signal,
        probability: 0.08,
        predicted_label: 'NORMAL',
      },
      decision: {
        ...baseResult.decision,
        manual_review_recommended: true,
        warning_code: 'MODEL_DISAGREEMENT',
      },
    })

    expect(
      screen.getByText('Models disagree'),
    ).toBeInTheDocument()

    expect(
      screen.getByText(
        'Manual review recommended',
      ),
    ).toBeInTheDocument()
  })

  it('shows explanation reliability warning when requested', () => {
    renderResult({
      explanation_quality: {
        ...baseResult.explanation_quality,
        quality_status: 'HIGH_SHORTCUT_RISK',
        display_warning: true,
        warning_code: 'HIGH_SHORTCUT_RISK',
        attribution_note:
          'Border attribution is elevated.',
      },
    })

    expect(
      screen.getByText(
        'Explanation reliability warning',
      ),
    ).toBeInTheDocument()

    expect(
      screen.getByText(
        'Border attribution is elevated.',
      ),
    ).toBeInTheDocument()
  })

  it('hides explanation reliability warning when not requested', () => {
    renderResult()

    expect(
      screen.queryByText(
        'Explanation reliability warning',
      ),
    ).not.toBeInTheDocument()
  })

  it('renders the unavailable overlay fallback', () => {
    render(
      <AnalysisResult
        result={baseResult}
        originalUrl="blob:original"
        overlayUrl={null}
      />,
    )

    expect(
      screen.getByText(
        'Explanation visualization unavailable.',
      ),
    ).toBeInTheDocument()
  })
})
