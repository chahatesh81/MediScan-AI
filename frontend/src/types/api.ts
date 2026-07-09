export type DiagnosisLabel =
  | 'NORMAL'
  | 'PNEUMONIA'

export type ExplanationMode =
  | 'positive_gradcam'
  | 'absolute_attribution'

export type QualityStatus =
  | 'HIGH_SHORTCUT_RISK'
  | 'ELEVATED_SHORTCUT_RISK'
  | 'LIMITED_SPATIAL_RELIABILITY'

export interface PrimaryPrediction {
  model: string
  label: DiagnosisLabel
  probability: number
  threshold: number
}

export interface SecondarySignal {
  model: string
  role: string
  probability: number
  threshold: number
  predicted_label: DiagnosisLabel
  automatic_override_allowed: boolean
}

export interface Decision {
  final_label: DiagnosisLabel
  source: string
  manual_review_recommended: boolean
  warning_code: string | null
}

export interface PreprocessingInfo {
  v1: string
  v3: string
  v3_metadata: Record<string, unknown>
}

export interface ExplanationInfo {
  method: 'gradcam'
  mode: ExplanationMode
  last_conv_layer: string
  raw_heatmap_shape: number[]
  output_width: number
  output_height: number
  minimum: number
  maximum: number
}

export interface AttentionQuality {
  border_energy_ratio: number
  thorax_energy_ratio: number
  peak_in_border: number
  quality_status: QualityStatus
  display_warning: boolean
  warning_code: string | null
  explanation_mode: ExplanationMode
  attribution_note: string | null
  region_definition: string
}

export interface VisualizationEndpoints {
  heatmap: '/api/v1/explain'
  overlay: '/api/v1/explain/overlay'
}

export interface AnalysisResponse {
  primary_prediction: PrimaryPrediction
  secondary_signal: SecondarySignal
  decision: Decision
  preprocessing: PreprocessingInfo
  explanation: ExplanationInfo
  explanation_quality: AttentionQuality
  visualization_endpoints: VisualizationEndpoints
  disclaimer: string
}

export type MedicalModuleStatus =
  | 'AVAILABLE'
  | 'PLANNED'

export type MedicalModality =
  | 'chest_xray'
  | 'brain_mri'
  | 'skin_image'
  | 'mammography'

export type ModuleTaskType =
  | 'binary_classification'
  | 'multiclass_classification'
  | 'multilabel_classification'

export interface ModuleDiscoveryItem {
  module_id: string
  display_name: string
  modality: MedicalModality
  task_type: ModuleTaskType
  status: MedicalModuleStatus
  output_classes: string[]
  supports_gradcam: boolean
  executable: boolean
}

export interface ModuleDiscoveryResponse {
  modules: ModuleDiscoveryItem[]
  total: number
}

export interface ModuleClassScore {
  label: string
  probability: number
}

export interface MulticlassModuleResult {
  result_type: 'multiclass_classification'
  predicted_label: string
  scores: ModuleClassScore[]
}

export interface ModuleAnalysisResponse {
  module_id: string
  display_name: string
  modality: MedicalModality
  task_type: ModuleTaskType
  result: MulticlassModuleResult
}
