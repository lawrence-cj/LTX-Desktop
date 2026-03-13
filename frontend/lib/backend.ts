let cached: { url: string; token: string } | null = null

export async function getBackendCredentials(): Promise<{ url: string; token: string }> {
  if (!cached) cached = await window.electronAPI.getBackend()
  return cached
}

export function resetBackendCredentials(): void {
  cached = null
}

export async function backendFetch(path: string, init?: RequestInit): Promise<Response> {
  const { url, token } = await getBackendCredentials()
  const headers = new Headers(init?.headers)
  if (token) headers.set('Authorization', `Bearer ${token}`)
  return fetch(`${url}${path}`, { ...init, headers })
}

/** Build an HTTP URL that serves an output file from the backend.
 *  Extracts the filename from a server-side absolute path and routes
 *  through the backend's /api/outputs/ endpoint so it works when the
 *  backend runs on a different machine (e.g. GPU cluster). */
export async function outputPathToUrl(serverPath: string): Promise<string> {
  const { url: backendUrl } = await getBackendCredentials()
  const filename = serverPath.replace(/\\/g, '/').split('/').pop()
  if (!filename) {
    const normalized = serverPath.replace(/\\/g, '/')
    return normalized.startsWith('/') ? `file://${normalized}` : `file:///${normalized}`
  }
  return `${backendUrl}/api/outputs/${encodeURIComponent(filename)}`
}

export async function backendWsUrl(path: string): Promise<string> {
  const { url, token } = await getBackendCredentials()
  const ws = url.replace('http://', 'ws://')
  const sep = path.includes('?') ? '&' : '?'
  return `${ws}${path}${sep}token=${token}`
}
