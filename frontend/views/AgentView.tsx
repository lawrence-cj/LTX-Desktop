import { useState, useEffect, useRef, useCallback } from 'react'
import { ArrowLeft, Play, Square, RefreshCw, Edit3, Film, Check, Loader2, PlayCircle, FolderOpen, ChevronUp, ChevronDown } from 'lucide-react'
import { Button } from '../components/ui/button'
import { Textarea } from '../components/ui/textarea'
import { useProjects } from '../contexts/ProjectContext'
import { backendFetch, outputPathToUrl } from '../lib/backend'
import type { Asset, TimelineClip, SubtitleClip } from '../types/project'
import { DEFAULT_COLOR_CORRECTION } from '../types/project'

// ============================================================
// Types
// ============================================================

type VideoStyle = 'cinematic' | 'anime' | 'realistic' | 'fantasy' | 'noir' | 'documentary'

interface ScenePlan {
  scene_index: number
  description: string
  prompt: string
  duration: number
  camera_motion: string
  image_path: string | null
}

interface AgentProgress {
  status: string
  current_scene: number
  total_scenes: number
  scene_status: string
  overall_progress: number
  completed_scenes: number[]
  failed_scenes: number[]
  scene_plans: ScenePlan[]
  video_paths: string[]
  final_video_path: string | null
  error: string | null
}

// ============================================================
// Constants
// ============================================================

const STYLES: { id: VideoStyle; label: string; desc: string }[] = [
  { id: 'cinematic', label: 'Cinematic', desc: 'Film-like, dramatic lighting' },
  { id: 'anime', label: 'Anime', desc: 'Japanese animation style' },
  { id: 'realistic', label: 'Realistic', desc: 'Photorealistic, natural' },
  { id: 'fantasy', label: 'Fantasy', desc: 'Magical, ethereal' },
  { id: 'noir', label: 'Noir', desc: 'High contrast, shadows' },
  { id: 'documentary', label: 'Documentary', desc: 'Raw, handheld feel' },
]

const RESOLUTIONS = ['720p', '1080p'] as const

const CAMERA_LABELS: Record<string, string> = {
  none: 'Auto',
  dolly_in: 'Push In',
  dolly_out: 'Pull Out',
  dolly_left: 'Track Left',
  dolly_right: 'Track Right',
  jib_up: 'Crane Up',
  jib_down: 'Crane Down',
  static: 'Static',
  focus_shift: 'Focus Shift',
}

// ============================================================
// Component
// ============================================================

export function AgentView() {
  const { goHome, createProject, addAsset, updateTimeline, getActiveTimeline, openProject, setCurrentTab } = useProjects()

  // Script input
  const [script, setScript] = useState('')
  const [style, setStyle] = useState<VideoStyle>('cinematic')
  const [resolution, setResolution] = useState<string>('720p')
  const [aspectRatio, setAspectRatio] = useState<'16:9' | '9:16'>('16:9')
  const [durationPerScene, setDurationPerScene] = useState(5)
  const [totalDuration, setTotalDuration] = useState(60)
  const [generateAudio, setGenerateAudio] = useState(false)
  const [parallelMode, setParallelMode] = useState(false)
  const [pipelineBackend, setPipelineBackend] = useState<string>('sana')
  const [serverBackend, setServerBackend] = useState<string | null>(null) // what the server actually loaded
  const [enableRefine, setEnableRefine] = useState(true)
  const [enableUpsample, setEnableUpsample] = useState(false)

  // Generation state
  const [isGenerating, setIsGenerating] = useState(false)
  const [progress, setProgress] = useState<AgentProgress | null>(null)
  const [error, setError] = useState<string | null>(null)

  // Plan preview (before generation)
  const [previewPlans, setPreviewPlans] = useState<ScenePlan[] | null>(null)
  const [isPlanLoading, setIsPlanLoading] = useState(false)

  // Scene editing & ordering
  const [editingScene, setEditingScene] = useState<number | null>(null)
  const [editedPrompts, setEditedPrompts] = useState<Record<number, string>>({})
  const [sceneImages, setSceneImages] = useState<Record<number, string>>({})
  const [sceneCameras, setSceneCameras] = useState<Record<number, string>>({})
  const [sceneOrder, setSceneOrder] = useState<number[] | null>(null) // null = default order

  // Scene video previews
  const [sceneVideoUrls, setSceneVideoUrls] = useState<Record<number, string>>({})
  const [playingScene, setPlayingScene] = useState<number | null>(null)
  const [finalVideoUrl, setFinalVideoUrl] = useState<string | null>(null)

  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null)
  const genStartTime = useRef<number>(0)

  // ============================================================
  // Polling
  // ============================================================

  const stopPolling = useCallback(() => {
    if (pollRef.current) {
      clearInterval(pollRef.current)
      pollRef.current = null
    }
  }, [])

  const startPolling = useCallback(() => {
    stopPolling()
    pollRef.current = setInterval(async () => {
      try {
        const res = await backendFetch('/api/agent/progress')
        if (!res.ok) return
        const data: AgentProgress = await res.json()
        setProgress(data)

        if (data.status === 'complete' || data.status === 'error' || data.status === 'cancelled') {
          stopPolling()
          setIsGenerating(false)
          if (data.error) setError(data.error)
        }
      } catch {
        // ignore polling errors
      }
    }, 1000)
  }, [stopPolling])

  useEffect(() => {
    return () => stopPolling()
  }, [stopPolling])

  // Fetch server pipeline backend on mount
  useEffect(() => {
    backendFetch('/api/runtime-policy').then(async res => {
      if (res.ok) {
        const data = await res.json()
        if (data.pipeline_backend) {
          setServerBackend(data.pipeline_backend)
          setPipelineBackend(data.pipeline_backend)
        }
      }
    }).catch(() => {})
  }, [])

  // On mount, check if there's an active generation to resume
  useEffect(() => {
    const checkActive = async () => {
      try {
        const res = await backendFetch('/api/agent/progress')
        if (!res.ok) return
        const data: AgentProgress = await res.json()
        if (data.status === 'generating' || data.status === 'planning' || data.status === 'assembling') {
          setIsGenerating(true)
          setProgress(data)
          genStartTime.current = Date.now() - (data.completed_scenes.length * 40 * 1000) // rough estimate
          startPolling()
        } else if (data.status === 'complete' && data.scene_plans.length > 0) {
          setProgress(data)
        }
      } catch { /* ignore */ }
    }
    checkActive()
  }, []) // eslint-disable-line react-hooks/exhaustive-deps

  // ============================================================
  // Generate
  // ============================================================

  const handleGenerate = async () => {
    if (!script.trim()) return

    setIsGenerating(true)
    setError(null)
    setProgress(null)
    genStartTime.current = Date.now()
    setSceneVideoUrls({})
    setFinalVideoUrl(null)
    setPlayingScene(null)

    // Build scenes from edited prompts if any
    const scenes = Object.keys(editedPrompts).length > 0
      ? Object.entries(editedPrompts).map(([_, desc]) => ({
          description: desc,
          duration: durationPerScene,
        }))
      : undefined

    try {
      const res = await backendFetch('/api/agent/generate/async', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          script: script.trim(),
          style,
          resolution,
          aspect_ratio: aspectRatio,
          duration_per_scene: durationPerScene,
          total_duration: totalDuration,
          fps: '24',
          model: 'fast',
          generate_audio: generateAudio,
          parallel: parallelMode,
          pipeline_backend: pipelineBackend,
          enable_refine: enableRefine,
          enable_upsample: enableUpsample,
          ...(scenes ? { scenes } : {}),
        }),
      })

      if (!res.ok) {
        const err = await res.json()
        throw new Error(err.error || 'Failed to start generation')
      }

      startPolling()
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Unknown error')
      setIsGenerating(false)
    }
  }

  const handlePreviewPlan = async () => {
    if (!script.trim()) return
    setIsPlanLoading(true)
    setError(null)
    try {
      const res = await backendFetch('/api/agent/plan', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          script: script.trim(),
          style,
          resolution,
          aspect_ratio: aspectRatio,
          duration_per_scene: durationPerScene,
          total_duration: totalDuration,
        }),
      })
      if (!res.ok) throw new Error('Failed to get plan')
      const data = await res.json()
      setPreviewPlans(data.scene_plans || [])
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Failed to preview plan')
    } finally {
      setIsPlanLoading(false)
    }
  }

  const handleCancel = async () => {
    try {
      await backendFetch('/api/agent/cancel', { method: 'POST' })
    } catch {
      // ignore
    }
  }

  // ============================================================
  // Scene regeneration
  // ============================================================

  const handleRegenerateScene = async (sceneIndex: number) => {
    if (!progress?.scene_plans?.[sceneIndex]) return

    const scene = progress.scene_plans[sceneIndex]
    const editedPrompt = editedPrompts[sceneIndex] || scene.description

    setIsGenerating(true)
    setError(null)

    try {
      const res = await backendFetch('/api/agent/generate/async', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          script: editedPrompt,
          style,
          resolution,
          aspect_ratio: aspectRatio,
          duration_per_scene: scene.duration,
          total_duration: scene.duration,
          fps: '24',
          model: 'fast',
          scenes: [{
            description: editedPrompt,
            duration: scene.duration,
            camera_motion: scene.camera_motion,
            image_path: scene.image_path,
          }],
        }),
      })

      if (!res.ok) throw new Error('Failed to regenerate scene')
      startPolling()
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Unknown error')
      setIsGenerating(false)
    }
  }

  // ============================================================
  // Open in Editor
  // ============================================================

  const handleOpenInEditor = useCallback(async () => {
    if (!progress?.video_paths || progress.video_paths.length === 0) return

    // Create a new project
    const projectName = `Agent: ${script.substring(0, 40)}${script.length > 40 ? '...' : ''}`
    const project = createProject(projectName)

    // Add each scene video as an asset and build timeline clips
    const clips: TimelineClip[] = []
    let currentTime = 0

    for (let i = 0; i < progress.video_paths.length; i++) {
      const videoPath = progress.video_paths[i]
      const scene = progress.scene_plans[i]
      let url: string
      try {
        url = await outputPathToUrl(videoPath)
      } catch {
        url = videoPath
      }

      const assetData: Omit<Asset, 'id' | 'createdAt'> = {
        type: 'video',
        path: videoPath,
        url,
        prompt: scene?.description || `Scene ${i + 1}`,
        resolution,
        duration: scene?.duration || durationPerScene,
      }
      const asset = addAsset(project.id, assetData)

      // Create a timeline clip for this asset
      const clipDuration = scene?.duration || durationPerScene
      clips.push({
        id: `clip-${Date.now()}-${i}`,
        assetId: asset.id,
        type: 'video',
        startTime: currentTime,
        duration: clipDuration,
        trimStart: 0,
        trimEnd: 0,
        speed: 1,
        reversed: false,
        muted: false,
        volume: 100,
        trackIndex: 0,
        asset,
        flipH: false,
        flipV: false,
        transitionIn: { type: 'none', duration: 0 },
        transitionOut: { type: i < progress.video_paths.length - 1 ? 'dissolve' : 'none', duration: 0.5 },
        colorCorrection: { ...DEFAULT_COLOR_CORRECTION },
        opacity: 100,
      })
      currentTime += clipDuration
    }

    // Build subtitle clips from scene descriptions
    const subtitles: SubtitleClip[] = []
    let subTime = 0
    for (let i = 0; i < progress.scene_plans.length; i++) {
      const scene = progress.scene_plans[i]
      const dur = scene?.duration || durationPerScene
      subtitles.push({
        id: `sub-${Date.now()}-${i}`,
        text: scene?.description || `Scene ${i + 1}`,
        startTime: subTime,
        endTime: subTime + dur,
        trackIndex: 0,
      })
      subTime += dur
    }

    // Update the timeline with clips and subtitles
    const timeline = getActiveTimeline(project.id)
    if (timeline) {
      updateTimeline(project.id, timeline.id, { clips, subtitles })
    }

    // Navigate to the project's video editor
    openProject(project.id)
    setCurrentTab('video-editor')
  }, [progress, script, resolution, durationPerScene, createProject, addAsset, updateTimeline, getActiveTimeline, openProject, setCurrentTab])

  // Load video URLs when progress shows completed scenes or final result
  useEffect(() => {
    if (!progress) return

    const loadUrls = async () => {
      // Load scene video URLs for completed scenes
      if (progress.video_paths && progress.video_paths.length > 0) {
        const urls: Record<number, string> = { ...sceneVideoUrls }
        let changed = false
        for (let i = 0; i < progress.video_paths.length; i++) {
          if (!urls[i]) {
            try {
              urls[i] = await outputPathToUrl(progress.video_paths[i])
              changed = true
            } catch { /* skip */ }
          }
        }
        if (changed) setSceneVideoUrls(urls)
      }

      // Load final video URL
      if (progress.final_video_path && !finalVideoUrl) {
        try {
          const url = await outputPathToUrl(progress.final_video_path)
          setFinalVideoUrl(url)
        } catch { /* skip */ }
      }
    }
    loadUrls()
  }, [progress?.completed_scenes?.length, progress?.status])

  // ============================================================
  // Render
  // ============================================================

  const rawScenePlans = progress?.scene_plans?.length ? progress.scene_plans : (previewPlans || [])
  const completedScenes = progress?.completed_scenes || []

  // Apply custom scene order if set (for reordering after generation)
  const scenePlans = sceneOrder
    ? sceneOrder.map(i => rawScenePlans[i]).filter(Boolean)
    : rawScenePlans

  const moveScene = (fromDisplayIdx: number, direction: 'up' | 'down') => {
    const order = sceneOrder || rawScenePlans.map((_, i) => i)
    const toDisplayIdx = direction === 'up' ? fromDisplayIdx - 1 : fromDisplayIdx + 1
    if (toDisplayIdx < 0 || toDisplayIdx >= order.length) return
    const newOrder = [...order]
    const tmp = newOrder[fromDisplayIdx]
    newOrder[fromDisplayIdx] = newOrder[toDisplayIdx]
    newOrder[toDisplayIdx] = tmp
    setSceneOrder(newOrder)
  }

  return (
    <div className="h-screen flex flex-col bg-zinc-950 text-white">
      {/* Header */}
      <div className="flex items-center gap-3 px-4 py-3 border-b border-zinc-800">
        <button onClick={goHome} className="text-zinc-400 hover:text-white transition-colors">
          <ArrowLeft size={20} />
        </button>
        <Film size={20} className="text-violet-400" />
        <h1 className="text-lg font-semibold">Video Agent</h1>
        <span className="text-xs text-zinc-500">Script to Video</span>
      </div>

      <div className="flex-1 flex overflow-hidden">
        {/* Left Panel: Script Input + Settings */}
        <div className="w-[400px] border-r border-zinc-800 flex flex-col overflow-y-auto">
          <div className="p-4 space-y-4">
            {/* Script Input */}
            <div>
              <label className="text-sm font-medium text-zinc-300 mb-2 block">Script</label>
              <Textarea
                value={script}
                onChange={(e) => setScript(e.target.value)}
                placeholder="Enter your video script... e.g. A robot walks through a neon-lit city at night, encounters a stray cat under a streetlight, and they watch the sunset together from a rooftop."
                className="min-h-[140px] bg-zinc-900 border-zinc-700 text-white placeholder:text-zinc-600 resize-none"
                disabled={isGenerating}
              />
              <div className="text-xs text-zinc-500 mt-1">
                {Math.ceil(totalDuration / durationPerScene)} scenes x {durationPerScene}s = ~{totalDuration}s total
              </div>
            </div>

            {/* Style Selection */}
            <div>
              <label className="text-sm font-medium text-zinc-300 mb-2 block">Style</label>
              <div className="grid grid-cols-3 gap-2">
                {STYLES.map(s => (
                  <button
                    key={s.id}
                    onClick={() => setStyle(s.id)}
                    disabled={isGenerating}
                    className={`p-2 rounded-lg border text-left transition-all ${
                      style === s.id
                        ? 'border-violet-500 bg-violet-500/10 text-violet-300'
                        : 'border-zinc-700 bg-zinc-900 text-zinc-400 hover:border-zinc-600'
                    }`}
                  >
                    <div className="text-xs font-medium">{s.label}</div>
                    <div className="text-[10px] text-zinc-500 mt-0.5">{s.desc}</div>
                  </button>
                ))}
              </div>
            </div>

            {/* Settings */}
            <div className="grid grid-cols-2 gap-3">
              <div>
                <label className="text-xs text-zinc-400 mb-1 block">Resolution</label>
                <select
                  value={resolution}
                  onChange={e => setResolution(e.target.value)}
                  disabled={isGenerating}
                  className="w-full bg-zinc-900 border border-zinc-700 rounded-md px-2 py-1.5 text-sm text-white"
                >
                  {RESOLUTIONS.map(r => <option key={r} value={r}>{r}</option>)}
                </select>
              </div>
              <div>
                <label className="text-xs text-zinc-400 mb-1 block">Scene Duration</label>
                <select
                  value={durationPerScene}
                  onChange={e => setDurationPerScene(Number(e.target.value))}
                  disabled={isGenerating}
                  className="w-full bg-zinc-900 border border-zinc-700 rounded-md px-2 py-1.5 text-sm text-white"
                >
                  {[3, 4, 5, 6, 8, 10].map(d => (
                    <option key={d} value={d}>{d}s</option>
                  ))}
                </select>
              </div>
              <div>
                <label className="text-xs text-zinc-400 mb-1 block">Total Duration</label>
                <select
                  value={totalDuration}
                  onChange={e => setTotalDuration(Number(e.target.value))}
                  disabled={isGenerating}
                  className="w-full bg-zinc-900 border border-zinc-700 rounded-md px-2 py-1.5 text-sm text-white"
                >
                  {[15, 30, 45, 60, 90, 120].map(d => (
                    <option key={d} value={d}>{d}s ({Math.ceil(d / durationPerScene)} scenes)</option>
                  ))}
                </select>
              </div>
              <div>
                <label className="text-xs text-zinc-400 mb-1 block">Aspect Ratio</label>
                <select
                  value={aspectRatio}
                  onChange={e => setAspectRatio(e.target.value as '16:9' | '9:16')}
                  disabled={isGenerating}
                  className="w-full bg-zinc-900 border border-zinc-700 rounded-md px-2 py-1.5 text-sm text-white"
                >
                  <option value="16:9">16:9</option>
                  <option value="9:16">9:16</option>
                </select>
              </div>
            </div>

            {/* Pipeline options */}
            <div className="space-y-2">
              <label className="text-sm font-medium text-zinc-300 block">Pipeline</label>
              <div className="grid grid-cols-3 gap-2">
                {(['ltx', 'sana', 'sana-diffusers'] as const).map(b => (
                  <button
                    key={b}
                    onClick={() => setPipelineBackend(b)}
                    disabled={isGenerating}
                    className={`px-2 py-1.5 rounded border text-xs transition-all ${
                      pipelineBackend === b
                        ? 'border-violet-500 bg-violet-500/10 text-violet-300'
                        : 'border-zinc-700 bg-zinc-900 text-zinc-400 hover:border-zinc-600'
                    }`}
                  >
                    {b === 'ltx' ? 'LTX 2.3' : b === 'sana' ? 'Sana+LTX' : 'Sana+Diffusers'}
                  </button>
                ))}
              </div>
              {serverBackend && pipelineBackend !== serverBackend && (
                <p className="text-[10px] text-amber-500">
                  Server loaded "{serverBackend}". Restart server with PIPELINE_BACKEND={pipelineBackend} to switch.
                </p>
              )}
              <div className="flex gap-4">
                <label className="flex items-center gap-2 cursor-pointer">
                  <input
                    type="checkbox"
                    checked={enableRefine}
                    onChange={e => setEnableRefine(e.target.checked)}
                    disabled={isGenerating || pipelineBackend === 'ltx'}
                    className="rounded border-zinc-600 bg-zinc-800 text-violet-500 focus:ring-violet-500"
                  />
                  <span className={`text-xs ${pipelineBackend === 'ltx' ? 'text-zinc-600' : 'text-zinc-400'}`}>Refiner</span>
                </label>
                <label className="flex items-center gap-2 cursor-pointer">
                  <input
                    type="checkbox"
                    checked={enableUpsample}
                    onChange={e => setEnableUpsample(e.target.checked)}
                    disabled={isGenerating}
                    className="rounded border-zinc-600 bg-zinc-800 text-violet-500 focus:ring-violet-500"
                  />
                  <span className="text-xs text-zinc-400">Upsampler (2x)</span>
                </label>
              </div>
            </div>

            {/* Audio toggle */}
            <label className="flex items-center gap-2 cursor-pointer">
              <input
                type="checkbox"
                checked={generateAudio}
                onChange={e => setGenerateAudio(e.target.checked)}
                disabled={isGenerating}
                className="rounded border-zinc-600 bg-zinc-800 text-violet-500 focus:ring-violet-500"
              />
              <span className="text-xs text-zinc-400">Generate audio for each scene</span>
            </label>
            <label className="flex items-center gap-2 cursor-pointer">
              <input
                type="checkbox"
                checked={parallelMode}
                onChange={e => setParallelMode(e.target.checked)}
                disabled={isGenerating}
                className="rounded border-zinc-600 bg-zinc-800 text-violet-500 focus:ring-violet-500"
              />
              <span className="text-xs text-zinc-400">Parallel generation (faster, no visual continuity)</span>
            </label>

            {/* Generate Button */}
            <div className="pt-2">
              {isGenerating ? (
                <Button
                  onClick={handleCancel}
                  variant="destructive"
                  className="w-full"
                >
                  <Square size={16} className="mr-2" />
                  Cancel Generation
                </Button>
              ) : (
                <Button
                  onClick={handleGenerate}
                  disabled={!script.trim()}
                  className="w-full bg-violet-600 hover:bg-violet-500"
                >
                  <Play size={16} className="mr-2" />
                  Generate Video
                </Button>
              )}
              {!isGenerating && (
                <Button
                  onClick={handlePreviewPlan}
                  disabled={!script.trim() || isPlanLoading}
                  variant="outline"
                  className="w-full mt-2 border-zinc-700"
                >
                  {isPlanLoading ? <Loader2 size={16} className="mr-2 animate-spin" /> : <Film size={16} className="mr-2" />}
                  Preview Plan
                </Button>
              )}
            </div>

            {/* Error */}
            {error && (
              <div className="p-3 rounded-lg bg-red-900/30 border border-red-800 text-red-300 text-sm">
                {error}
              </div>
            )}
          </div>
        </div>

        {/* Right Panel: Progress + Scene List + Preview */}
        <div className="flex-1 flex flex-col overflow-hidden">
          {/* Progress Bar */}
          {(isGenerating || progress) && (() => {
            // Estimate remaining time
            const elapsed = (Date.now() - genStartTime.current) / 1000
            const completedCount = progress?.completed_scenes?.length || 0
            const totalCount = progress?.total_scenes || 0
            let timeEstimate = ''
            if (completedCount > 0 && totalCount > completedCount && elapsed > 0) {
              const avgPerScene = elapsed / completedCount
              const remaining = (totalCount - completedCount) * avgPerScene
              const mins = Math.ceil(remaining / 60)
              timeEstimate = mins > 1 ? `~${mins} min remaining` : `< 1 min remaining`
            } else if (isGenerating && totalCount > 0 && completedCount === 0) {
              // First scene is loading model, rough estimate
              timeEstimate = `Loading model + ${totalCount} scenes...`
            }
            return (
            <div className="px-4 py-3 border-b border-zinc-800">
              <div className="flex items-center justify-between mb-2">
                <span className="text-sm text-zinc-300">
                  {progress?.scene_status || 'Starting...'}
                </span>
                <div className="flex items-center gap-3">
                  {timeEstimate && <span className="text-xs text-zinc-500">{timeEstimate}</span>}
                  <span className="text-sm text-zinc-400">
                    {progress?.overall_progress || 0}%
                  </span>
                </div>
              </div>
              <div className="h-2 bg-zinc-800 rounded-full overflow-hidden">
                <div
                  className="h-full bg-violet-500 rounded-full transition-all duration-500"
                  style={{ width: `${progress?.overall_progress || 0}%` }}
                />
              </div>
            </div>
            )
          })()}

          <div className="flex-1 flex overflow-hidden">
            {/* Scene Grid */}
            <div className="flex-1 overflow-y-auto p-4">
              {scenePlans.length > 0 ? (
                <div className="space-y-3">
                  <h2 className="text-sm font-medium text-zinc-400 uppercase tracking-wider">
                    Scenes ({scenePlans.length})
                  </h2>
                  <div className="grid grid-cols-1 lg:grid-cols-2 gap-3">
                    {scenePlans.map((scene, idx) => {
                      const isCompleted = completedScenes.includes(idx)
                      const isFailed = (progress?.failed_scenes || []).includes(idx)
                      const isCurrent = progress?.current_scene === idx && progress?.status === 'generating'
                      const isEditing = editingScene === idx
                      const hasVideo = sceneVideoUrls[idx] != null

                      return (
                        <div
                          key={idx}
                          className={`rounded-lg border p-3 transition-all ${
                            isFailed
                              ? 'border-red-700 bg-red-900/10'
                              : isCurrent
                              ? 'border-violet-500 bg-violet-500/5'
                              : isCompleted
                              ? 'border-green-700 bg-green-900/10'
                              : 'border-zinc-800 bg-zinc-900'
                          }`}
                        >
                          {/* Scene video thumbnail */}
                          {hasVideo && (
                            <div
                              className="relative mb-2 rounded overflow-hidden bg-black cursor-pointer group"
                              onClick={() => setPlayingScene(playingScene === idx ? null : idx)}
                            >
                              {playingScene === idx ? (
                                <video
                                  src={sceneVideoUrls[idx]}
                                  autoPlay
                                  loop
                                  muted
                                  className="w-full aspect-video object-cover"
                                />
                              ) : (
                                <div className="w-full aspect-video bg-zinc-800 flex items-center justify-center">
                                  <PlayCircle size={32} className="text-zinc-500 group-hover:text-violet-400 transition-colors" />
                                </div>
                              )}
                            </div>
                          )}

                          <div className="flex items-start justify-between gap-2">
                            <div className="flex items-center gap-2 min-w-0">
                              <span className={`text-xs font-mono px-1.5 py-0.5 rounded flex items-center ${
                                isFailed ? 'bg-red-800 text-red-200' :
                                isCompleted ? 'bg-green-800 text-green-200' :
                                isCurrent ? 'bg-violet-800 text-violet-200' :
                                'bg-zinc-800 text-zinc-400'
                              }`}>
                                {isFailed ? 'FAIL' :
                                 isCompleted ? <Check size={12} /> :
                                 isCurrent ? <Loader2 size={12} className="animate-spin" /> :
                                 `#${idx + 1}`}
                              </span>
                              <span className="text-xs text-zinc-500">{scene.duration}s</span>
                              <span className="text-xs text-zinc-600">
                                {CAMERA_LABELS[scene.camera_motion] || scene.camera_motion}
                              </span>
                            </div>
                            <div className="flex items-center gap-1">
                              <button
                                onClick={() => setEditingScene(isEditing ? null : idx)}
                                className="p-1 text-zinc-500 hover:text-zinc-300 transition-colors"
                                title="Edit prompt"
                              >
                                <Edit3 size={14} />
                              </button>
                              {(isCompleted || isFailed) && !isGenerating && (
                                <>
                                  <button
                                    onClick={() => handleRegenerateScene(idx)}
                                    className={`p-1 transition-colors ${isFailed ? 'text-red-400 hover:text-red-300' : 'text-zinc-500 hover:text-violet-400'}`}
                                    title={isFailed ? 'Retry this scene' : 'Regenerate this scene'}
                                  >
                                    <RefreshCw size={14} />
                                  </button>
                                  <button
                                    onClick={() => moveScene(idx, 'up')}
                                    disabled={idx === 0}
                                    className="p-1 text-zinc-500 hover:text-zinc-300 disabled:opacity-30 transition-colors"
                                    title="Move up"
                                  >
                                    <ChevronUp size={14} />
                                  </button>
                                  <button
                                    onClick={() => moveScene(idx, 'down')}
                                    disabled={idx === scenePlans.length - 1}
                                    className="p-1 text-zinc-500 hover:text-zinc-300 disabled:opacity-30 transition-colors"
                                    title="Move down"
                                  >
                                    <ChevronDown size={14} />
                                  </button>
                                </>
                              )}
                            </div>
                          </div>

                          {isEditing ? (
                            <div className="mt-2 space-y-2">
                              <Textarea
                                value={editedPrompts[idx] ?? scene.description}
                                onChange={e => setEditedPrompts(prev => ({ ...prev, [idx]: e.target.value }))}
                                className="text-xs bg-zinc-800 border-zinc-700 min-h-[60px]"
                              />
                              {/* Image conditioning for I2V */}
                              <div className="flex items-center gap-2">
                                <label className="text-[10px] text-zinc-500 cursor-pointer hover:text-zinc-300 transition-colors flex items-center gap-1">
                                  <input
                                    type="file"
                                    accept="image/*"
                                    className="hidden"
                                    onChange={e => {
                                      const file = e.target.files?.[0]
                                      if (file) {
                                        const url = URL.createObjectURL(file)
                                        setSceneImages(prev => ({ ...prev, [idx]: url }))
                                      }
                                    }}
                                  />
                                  {sceneImages[idx] ? 'Change image' : '+ Add image (I2V)'}
                                </label>
                                {sceneImages[idx] && (
                                  <img src={sceneImages[idx]} alt="" className="h-8 w-12 object-cover rounded" />
                                )}
                              </div>
                              {/* Camera motion override */}
                              <div className="flex items-center gap-2">
                                <span className="text-[10px] text-zinc-500">Camera:</span>
                                <select
                                  value={sceneCameras[idx] ?? scene.camera_motion}
                                  onChange={e => setSceneCameras(prev => ({ ...prev, [idx]: e.target.value }))}
                                  className="bg-zinc-800 border border-zinc-700 rounded px-1.5 py-0.5 text-[10px] text-zinc-300"
                                >
                                  {Object.entries(CAMERA_LABELS).map(([val, label]) => (
                                    <option key={val} value={val}>{label}</option>
                                  ))}
                                </select>
                              </div>
                            </div>
                          ) : (
                            <p className="mt-2 text-xs text-zinc-400 line-clamp-2">
                              {editedPrompts[idx] ?? scene.description}
                            </p>
                          )}

                          {/* Prompt preview */}
                          <p className="mt-1 text-[10px] text-zinc-600 line-clamp-1 italic">
                            {scene.prompt}
                          </p>
                        </div>
                      )
                    })}
                  </div>
                </div>
              ) : !isGenerating ? (
                <div className="flex flex-col items-center justify-center h-full text-zinc-500">
                  <Film size={48} className="mb-4 text-zinc-700" />
                  <p className="text-lg mb-2">Enter a script to get started</p>
                  <p className="text-sm text-zinc-600 max-w-md text-center">
                    The Video Agent will automatically decompose your script into scenes,
                    generate each scene using LTX-Video, and assemble them into a final video.
                  </p>
                </div>
              ) : null}
            </div>

            {/* Right side: Final video preview */}
            {finalVideoUrl && (
              <div className="w-[380px] border-l border-zinc-800 flex flex-col overflow-y-auto p-4">
                <h2 className="text-sm font-medium text-zinc-400 uppercase tracking-wider mb-3">
                  Final Video
                </h2>
                <div className="rounded-lg overflow-hidden border border-zinc-700 bg-black">
                  <video
                    src={finalVideoUrl}
                    controls
                    className="w-full"
                    autoPlay
                    loop
                  />
                </div>
                <p className="text-xs text-zinc-600 mt-2">
                  {scenePlans.length} scenes, {scenePlans.reduce((acc, s) => acc + s.duration, 0)}s total
                </p>
                <Button
                  onClick={handleOpenInEditor}
                  className="mt-3 w-full bg-blue-600 hover:bg-blue-500"
                  disabled={!progress?.video_paths?.length}
                >
                  <FolderOpen size={16} className="mr-2" />
                  Open in Video Editor
                </Button>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  )
}
