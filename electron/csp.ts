import { session } from 'electron'
import { isDev } from './config'
import { getAuthToken, getBackendUrl } from './python-backend'

// Enforce Content Security Policy via response headers (tamper-proof from renderer)
export function setupCSP(): void {
  session.defaultSession.webRequest.onHeadersReceived((details, callback) => {
    const csp = isDev
      ? [
          "default-src 'self'",
          "script-src 'self' 'unsafe-inline'",
          "style-src 'self' 'unsafe-inline' https://fonts.googleapis.com",
          "font-src 'self' https://fonts.gstatic.com",
          "connect-src 'self' http://localhost:* http://127.0.0.1:* ws://localhost:* ws://127.0.0.1:*",
          "img-src 'self' data: blob: file: http://localhost:* http://127.0.0.1:*",
          "media-src 'self' blob: file: http://localhost:* http://127.0.0.1:*",
          "object-src 'none'",
          "base-uri 'self'",
          "form-action 'self'",
          "frame-ancestors 'none'",
        ].join('; ')
      : [
          "default-src 'self'",
          "script-src 'self'",
          "style-src 'self' https://fonts.googleapis.com",
          "font-src 'self' https://fonts.gstatic.com",
          "connect-src 'self' http://localhost:* http://127.0.0.1:* ws://localhost:* ws://127.0.0.1:*",
          "img-src 'self' data: blob: file: http://localhost:* http://127.0.0.1:*",
          "media-src 'self' blob: file: http://localhost:* http://127.0.0.1:*",
          "object-src 'none'",
          "base-uri 'self'",
          "form-action 'self'",
          "frame-ancestors 'none'",
        ].join('; ')

    callback({
      responseHeaders: {
        ...details.responseHeaders,
        'Content-Security-Policy': [csp],
      },
    })
  })

  // Inject auth token into requests to the backend so that <img src> and
  // <video src> (which cannot attach headers themselves) pass authentication.
  session.defaultSession.webRequest.onBeforeSendHeaders(
    { urls: ['http://localhost:*/*', 'http://127.0.0.1:*/*'] },
    (details, callback) => {
      const token = getAuthToken()
      const backendUrl = getBackendUrl()

      if (token && backendUrl && details.url.startsWith(backendUrl)) {
        details.requestHeaders['Authorization'] = `Bearer ${token}`
      }

      callback({ requestHeaders: details.requestHeaders })
    },
  )
}
