import {
  afterEach,
  describe,
  expect,
  it,
  vi,
} from 'vitest'

import {
  MediscanApiError,
  analyzeImage,
  fetchExplanationOverlay,
} from '../mediscan'

const VALID_ANALYSIS = {
  primary_prediction: {
    model: 'baseline_cnn_v1',
    label: 'PNEUMONIA',
    probability: 0.998275279999,
    threshold: 0.053276624531,
  },
  secondary_signal: {
    model: 'advanced_v3',
    role: 'exploratory',
    probability: 0.92649692297,
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
    v1: 'rgb_bilinear_resize_224',
    v3: 'artifact_aware_preprocess_xray',
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
    border_energy_ratio: 0.619394,
    thorax_energy_ratio: 0.46559,
    peak_in_border: 1,
    quality_status: 'ELEVATED_SHORTCUT_RISK',
    display_warning: true,
    warning_code:
      'EXPLANATION_ELEVATED_SHORTCUT_RISK',
    explanation_mode: 'absolute_attribution',
    attribution_note: null,
    region_definition: 'test-region',
  },
  visualization_endpoints: {
    heatmap: '/api/v1/explain',
    overlay: '/api/v1/explain/overlay',
  },
  disclaimer: 'Not for clinical use.',
}

function createFile() {
  return new File(
    ['x-ray-bytes'],
    'test-xray.jpeg',
    {
      type: 'image/jpeg',
    },
  )
}

afterEach(() => {
  vi.unstubAllGlobals()
})

describe('analyzeImage', () => {
  it('returns a valid analysis response', async () => {
    const fetchMock = vi.fn().mockResolvedValue(
      new Response(
        JSON.stringify(VALID_ANALYSIS),
        {
          status: 200,
          headers: {
            'content-type': 'application/json',
          },
        },
      ),
    )

    vi.stubGlobal('fetch', fetchMock)

    const result = await analyzeImage(createFile())

    expect(
      result.primary_prediction.model,
    ).toBe('baseline_cnn_v1')

    expect(
      result.decision.final_label,
    ).toBe('PNEUMONIA')

    expect(fetchMock).toHaveBeenCalledOnce()

    expect(fetchMock).toHaveBeenCalledWith(
      '/api/v1/analyze',
      expect.objectContaining({
        method: 'POST',
        body: expect.any(FormData),
      }),
    )
  })

  it('preserves backend validation errors', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn().mockResolvedValue(
        new Response(
          JSON.stringify({
            detail:
              'Unsupported image type. Use JPEG or PNG.',
          }),
          {
            status: 415,
            headers: {
              'content-type': 'application/json',
            },
          },
        ),
      ),
    )

    await expect(
      analyzeImage(createFile()),
    ).rejects.toMatchObject({
      name: 'MediscanApiError',
      kind: 'UNSUPPORTED_TYPE',
      status: 415,
      message:
        'Unsupported image type. Use JPEG or PNG.',
    })
  })

  it('classifies network failures', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn().mockRejectedValue(
        new TypeError('Failed to fetch'),
      ),
    )

    await expect(
      analyzeImage(createFile()),
    ).rejects.toMatchObject({
      name: 'MediscanApiError',
      kind: 'NETWORK_ERROR',
      status: null,
    })
  })

  it('rejects invalid JSON responses', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn().mockResolvedValue(
        new Response(
          'not-json',
          {
            status: 200,
          },
        ),
      ),
    )

    await expect(
      analyzeImage(createFile()),
    ).rejects.toMatchObject({
      name: 'MediscanApiError',
      kind: 'INVALID_RESPONSE',
      status: 200,
    })
  })
})

describe('fetchExplanationOverlay', () => {
  it('returns an object URL for a PNG', async () => {
    const createObjectURL = vi.fn(
      () => 'blob:mediscan-overlay',
    )

    vi.stubGlobal(
      'fetch',
      vi.fn().mockResolvedValue(
        new Response(
          new Blob(
            ['png-bytes'],
            {
              type: 'image/png',
            },
          ),
          {
            status: 200,
            headers: {
              'content-type': 'image/png',
            },
          },
        ),
      ),
    )

    vi.stubGlobal(
      'URL',
      {
        ...URL,
        createObjectURL,
      },
    )

    const result =
      await fetchExplanationOverlay(createFile())

    expect(result).toBe(
      'blob:mediscan-overlay',
    )

    expect(createObjectURL).toHaveBeenCalledOnce()
  })

  it('rejects a non-PNG overlay response', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn().mockResolvedValue(
        new Response(
          'not-an-image',
          {
            status: 200,
            headers: {
              'content-type': 'text/plain',
            },
          },
        ),
      ),
    )

    await expect(
      fetchExplanationOverlay(createFile()),
    ).rejects.toBeInstanceOf(
      MediscanApiError,
    )

    await expect(
      fetchExplanationOverlay(createFile()),
    ).rejects.toMatchObject({
      kind: 'INVALID_RESPONSE',
      status: 200,
    })
  })
})
