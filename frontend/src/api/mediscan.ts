import type { AnalysisResponse } from '../types/api'

interface ApiErrorBody {
  detail?: string
}

export type ApiErrorKind =
  | 'BAD_REQUEST'
  | 'TOO_LARGE'
  | 'UNSUPPORTED_TYPE'
  | 'SERVER_ERROR'
  | 'NETWORK_ERROR'
  | 'INVALID_RESPONSE'
  | 'UNKNOWN_ERROR'

export class MediscanApiError extends Error {
  readonly kind: ApiErrorKind
  readonly status: number | null

  constructor(
    message: string,
    kind: ApiErrorKind,
    status: number | null = null,
  ) {
    super(message)
    this.name = 'MediscanApiError'
    this.kind = kind
    this.status = status
  }
}

async function getErrorMessage(
  response: Response,
): Promise<string | null> {
  try {
    const body = (await response.json()) as ApiErrorBody

    if (
      typeof body.detail === 'string'
      && body.detail.trim()
    ) {
      return body.detail
    }
  } catch {
    // The response did not contain valid JSON.
  }

  return null
}

function getErrorKind(status: number): ApiErrorKind {
  if (status === 400) {
    return 'BAD_REQUEST'
  }

  if (status === 413) {
    return 'TOO_LARGE'
  }

  if (status === 415) {
    return 'UNSUPPORTED_TYPE'
  }

  if (status >= 500) {
    return 'SERVER_ERROR'
  }

  return 'UNKNOWN_ERROR'
}

async function throwResponseError(
  response: Response,
): Promise<never> {
  const backendMessage = await getErrorMessage(response)

  throw new MediscanApiError(
    backendMessage
      ?? `Request failed with status ${response.status}.`,
    getErrorKind(response.status),
    response.status,
  )
}

function createImageFormData(file: File): FormData {
  const formData = new FormData()
  formData.append('file', file)

  return formData
}

async function request(
  endpoint: string,
  file: File,
): Promise<Response> {
  try {
    return await fetch(endpoint, {
      method: 'POST',
      body: createImageFormData(file),
    })
  } catch {
    throw new MediscanApiError(
      'Could not connect to the MediScan backend. Check that the API server is running and try again.',
      'NETWORK_ERROR',
    )
  }
}

export async function analyzeImage(
  file: File,
): Promise<AnalysisResponse> {
  const response = await request(
    '/api/v1/analyze',
    file,
  )

  if (!response.ok) {
    await throwResponseError(response)
  }

  try {
    return (await response.json()) as AnalysisResponse
  } catch {
    throw new MediscanApiError(
      'The backend returned an invalid analysis response.',
      'INVALID_RESPONSE',
      response.status,
    )
  }
}

export async function fetchExplanationOverlay(
  file: File,
): Promise<string> {
  const response = await request(
    '/api/v1/explain/overlay',
    file,
  )

  if (!response.ok) {
    await throwResponseError(response)
  }

  const contentType =
    response.headers.get('content-type') ?? ''

  if (!contentType.startsWith('image/png')) {
    throw new MediscanApiError(
      'The backend returned an invalid explanation image.',
      'INVALID_RESPONSE',
      response.status,
    )
  }

  const imageBlob = await response.blob()

  if (imageBlob.size === 0) {
    throw new MediscanApiError(
      'The backend returned an empty explanation image.',
      'INVALID_RESPONSE',
      response.status,
    )
  }

  return URL.createObjectURL(imageBlob)
}
