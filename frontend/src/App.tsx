import {
  LoaderCircle,
  ScanSearch,
} from 'lucide-react'
import {
  useEffect,
  useRef,
  useState,
} from 'react'
import {
  analyzeImage,
  fetchExplanationOverlay,
} from './api/mediscan'
import AnalysisResult from './components/AnalysisResult'
import UploadZone from './components/UploadZone'
import WorkflowProgress, {
  type WorkflowStage,
} from './components/WorkflowProgress'
import WorkspaceHeader from './components/WorkspaceHeader'
import type { AnalysisResponse } from './types/api'

function App() {
  const [file, setFile] = useState<File | null>(null)
  const [previewUrl, setPreviewUrl] =
    useState<string | null>(null)
  const [overlayUrl, setOverlayUrl] =
    useState<string | null>(null)
  const [result, setResult] =
    useState<AnalysisResponse | null>(null)
  const [error, setError] =
    useState<string | null>(null)
  const [overlayWarning, setOverlayWarning] =
    useState<string | null>(null)
  const [isAnalyzing, setIsAnalyzing] =
    useState(false)
  const requestIdRef = useRef(0)

  const workflowStage: WorkflowStage = result
    ? 'complete'
    : isAnalyzing
      ? 'analyzing'
      : file
        ? 'ready'
        : 'upload'

  useEffect(() => {
    return () => {
      if (previewUrl) {
        URL.revokeObjectURL(previewUrl)
      }

      if (overlayUrl) {
        URL.revokeObjectURL(overlayUrl)
      }
    }
  }, [previewUrl, overlayUrl])

  function clearAnalysis() {
    if (previewUrl) {
      URL.revokeObjectURL(previewUrl)
    }

    if (overlayUrl) {
      URL.revokeObjectURL(overlayUrl)
    }

    setFile(null)
    setPreviewUrl(null)
    setOverlayUrl(null)
    setResult(null)
    setError(null)
    setOverlayWarning(null)
    requestIdRef.current += 1
  }

  function handleFileSelect(selectedFile: File) {
    if (previewUrl) {
      URL.revokeObjectURL(previewUrl)
    }

    if (overlayUrl) {
      URL.revokeObjectURL(overlayUrl)
    }

    setFile(selectedFile)
    setPreviewUrl(
      URL.createObjectURL(selectedFile),
    )
    setOverlayUrl(null)
    setResult(null)
    setError(null)
    setOverlayWarning(null)
    requestIdRef.current += 1
  }

  async function handleAnalyze() {
    if (!file || isAnalyzing) {
      return
    }

    const requestId = requestIdRef.current + 1
    requestIdRef.current = requestId

    setIsAnalyzing(true)
    setError(null)
    setOverlayWarning(null)
    setResult(null)

    if (overlayUrl) {
      URL.revokeObjectURL(overlayUrl)
      setOverlayUrl(null)
    }

    try {
      const analysis = await analyzeImage(file)

      if (requestIdRef.current !== requestId) {
        return
      }

      setResult(analysis)

      try {
        const explanationOverlay =
          await fetchExplanationOverlay(file)

        if (requestIdRef.current !== requestId) {
          URL.revokeObjectURL(
            explanationOverlay,
          )
          return
        }

        setOverlayUrl(explanationOverlay)
      } catch {
        if (requestIdRef.current === requestId) {
          setOverlayWarning(
            'Analysis completed successfully, but the Grad-CAM overlay could not be loaded.',
          )
        }
      }
    } catch (caughtError) {
      if (requestIdRef.current !== requestId) {
        return
      }

      setError(
        caughtError instanceof Error
          ? caughtError.message
          : 'Analysis could not be completed.',
      )
    } finally {
      if (requestIdRef.current === requestId) {
        setIsAnalyzing(false)
      }
    }
  }

  return (
    <main className="min-h-screen bg-slate-950 text-slate-100">
      <WorkspaceHeader />

      <div className="mx-auto w-full max-w-7xl px-4 py-8 sm:px-6 lg:px-8">
        <WorkflowProgress stage={workflowStage} />

        <section className="mt-10 max-w-4xl">
          <div className="inline-flex items-center gap-2 rounded-full border border-cyan-500/20 bg-cyan-500/10 px-4 py-2 text-xs font-medium text-cyan-300">
            <ScanSearch className="size-4" />
            AI-assisted decision-support prototype
          </div>

          <h1 className="mt-6 text-4xl font-semibold tracking-tight sm:text-6xl">
            Analyze a chest X-ray
          </h1>

          <p className="mt-5 max-w-2xl text-base leading-7 text-slate-400 sm:text-lg">
            Run the validated V1 classifier, inspect the
            exploratory V3 safety signal, and review the
            Grad-CAM explanation with attention-quality
            diagnostics.
          </p>
        </section>

        <section className="mt-10 w-full">
          <UploadZone
            file={file}
            previewUrl={previewUrl}
            disabled={isAnalyzing}
            onFileSelect={handleFileSelect}
            onClear={clearAnalysis}
            onValidationError={setError}
          />

          {file && !result && (
            <button
              type="button"
              onClick={handleAnalyze}
              disabled={isAnalyzing}
              className="mt-5 flex w-full items-center justify-center gap-3 rounded-2xl bg-cyan-400 px-6 py-4 font-semibold text-slate-950 transition hover:bg-cyan-300 disabled:cursor-wait disabled:opacity-70"
            >
              {isAnalyzing ? (
                <>
                  <LoaderCircle className="size-5 animate-spin" />
                  Running AI analysis...
                </>
              ) : (
                <>
                  <ScanSearch className="size-5" />
                  Analyze X-ray
                </>
              )}
            </button>
          )}

          {error && (
            <div
              role="alert"
              className="mt-5 rounded-2xl border border-rose-500/30 bg-rose-500/10 p-5 text-sm text-rose-200"
            >
              {error}
            </div>
          )}

          {overlayWarning && (
            <div
              role="status"
              className="mt-5 rounded-2xl border border-amber-500/30 bg-amber-500/10 p-5 text-sm text-amber-200"
            >
              {overlayWarning}
            </div>
          )}

          {result && previewUrl && (
            <>
              <AnalysisResult
                result={result}
                originalUrl={previewUrl}
                overlayUrl={overlayUrl}
              />

              <button
                type="button"
                onClick={clearAnalysis}
                className="mt-6 w-full rounded-2xl border border-slate-700 px-6 py-4 font-medium text-slate-300 transition hover:border-slate-600 hover:bg-slate-900"
              >
                Analyze another X-ray
              </button>
            </>
          )}
        </section>

        <footer className="mt-16 border-t border-slate-900 py-8 text-sm leading-6 text-slate-500">
          Educational decision-support prototype. Not for
          clinical use. Human review required.
        </footer>
      </div>
    </main>
  )
}

export default App
