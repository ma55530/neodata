import { useEffect, useMemo, useState } from "react"
import { Link } from "react-router-dom"
import heic2any from "heic2any"
import "./App.css"
import NeoDataLogo from "./assets/neodata.jpg"

const API_BASE_URL = process.env.REACT_APP_API_BASE_URL || "http://localhost:8000"
const COMBINED_ENDPOINT = `${API_BASE_URL}/detect/combined`
const SEGMENTATION_BASE_URL = `${API_BASE_URL}/segmentation`
const KFK_LOGO_URL = "https://www.kfk.hr/static/uploads/2020/10/logo-light.png"
const COMMINUS_LOGO_URL = "https://www.comminus.hr/wp-content/uploads/2025/06/logo_400x200.png"

function App() {
  const [file, setFile] = useState(null)
  const [previewUrl, setPreviewUrl] = useState("")
  const [prediction, setPrediction] = useState(null)
  const [status, setStatus] = useState("idle")
  const [error, setError] = useState("")
  const [history, setHistory] = useState([])

  const isPassingLabel = (label) => {
    if (!label) {
      return false
    }
    const normalized = label.toLowerCase()
    return normalized === "good" || normalized === "pass" || normalized === "positive"
  }

  const formatPercent = (value, fallback = "—") => {
    if (typeof value !== "number" || Number.isNaN(value)) {
      return fallback
    }
    return `${value.toFixed(1)}%`
  }

  useEffect(() => {
    return () => {
      if (previewUrl) {
        URL.revokeObjectURL(previewUrl)
      }
    }
  }, [previewUrl])

  const convertHeicViaBackend = async (inputFile) => {
    const body = new FormData()
    body.append("file", inputFile)

    const response = await fetch(`${API_BASE_URL}/convert/heic`, {
      method: "POST",
      body,
    })

    if (!response.ok) {
      let detail = "Server nije mogao pretvoriti HEIC datoteku."
      const rawMessage = (await response.text()).trim()
      if (rawMessage) {
        try {
          const parsed = JSON.parse(rawMessage)
          detail = parsed.detail || parsed.message || detail
        } catch {
          detail = rawMessage
        }
      }
      throw new Error(detail)
    }

    const blob = await response.blob()
    const sanitizedName = (inputFile.name || "converted").replace(/\.hei[cf]$/i, "") || "converted"
    return new File([blob], `${sanitizedName}.jpg`, { type: blob.type || "image/jpeg" })
  }

  const convertFileIfNeeded = async (inputFile) => {
    const isHeic = /image\/hei[cf]/i.test(inputFile.type) || /\.hei[cf]$/i.test(inputFile.name)
    if (!isHeic) {
      return inputFile
    }

    try {
      const convertedBlob = await heic2any({
        blob: inputFile,
        toType: "image/jpeg",
        quality: 0.9,
      })

      return new File([convertedBlob], inputFile.name.replace(/\.hei[cf]$/i, ".jpg"), {
        type: "image/jpeg",
      })
    } catch (clientError) {
      console.warn("Local HEIC conversion failed, using backend fallback.", clientError)
      const fallbackResult = await convertHeicViaBackend(inputFile)
      return fallbackResult
    }
  }

  const handleFileChange = async (event) => {
    const selectedFile = event.target.files?.[0]
    if (!selectedFile) {
      return
    }

    try {
      const normalizedFile = await convertFileIfNeeded(selectedFile)
      if (previewUrl) {
        URL.revokeObjectURL(previewUrl)
      }
      const nextPreview = URL.createObjectURL(normalizedFile)
      setFile(normalizedFile)
      setPrediction(null)
      setError("")
      setPreviewUrl(nextPreview)
    } catch (conversionError) {
      console.error("HEIC conversion failed", conversionError)
      setFile(null)
      setPreviewUrl("")
      if (conversionError instanceof Error && conversionError.message) {
        setError(conversionError.message)
      } else {
        setError("HEIC format nije podržan za prikaz. Pretvorba nije uspjela.")
      }
    }
  }

  const handleReset = () => {
    setFile(null)
    setPreviewUrl("")
    setPrediction(null)
    setError("")
    setStatus("idle")
  }

  const handleSubmit = async () => {
    if (!file) {
      setError("Dodajte fotografiju za analizirati")
      return
    }

    setStatus("processing")
    setError("")

    const body = new FormData()
    body.append("file", file)

    try {
      const response = await fetch(COMBINED_ENDPOINT, {
        method: "POST",
        body,
      })

      let payload
      try {
        payload = await response.json()
      } catch (parseError) {
        throw new Error("Ne mogu pročitati odgovor modela.")
      }

      if (!response.ok) {
        throw new Error(payload.detail || "Model je odbio zahtjev.")
      }

      setPrediction(payload)
      setHistory((prev) => [{ ...payload, timestamp: Date.now() }, ...prev].slice(0, 4))
      setStatus("done")
    } catch (requestError) {
      setError(requestError.message || "Dogodila se neočekivana greška.")
      setStatus("error")
    }
  }

  const confidencePercent = useMemo(() => {
    if (!prediction?.confidence) {
      return null
    }
    return `${(prediction.confidence * 100).toFixed(1)}%`
  }, [prediction])

  // --- Download helpers for current prediction report ---
  const downloadBlob = (content, filename, type) => {
    const blob = new Blob([content], { type })
    const url = URL.createObjectURL(blob)
    const a = document.createElement("a")
    a.href = url
    a.download = filename
    document.body.appendChild(a)
    a.click()
    a.remove()
    URL.revokeObjectURL(url)
  }

  const downloadReportJSON = async () => {
    if (!prediction) return
    const base = (prediction.original_jpeg || "report").replace(/\s+/g, "_")
    const filename = `${base}.json`

    // try to fetch backend /health and append
    let health = null
    try {
      const res = await fetch(`${API_BASE_URL}/health`)
      if (res.ok) {
        health = await res.json()
      } else {
        health = { error: `health endpoint returned ${res.status}` }
      }
    } catch (e) {
      health = { error: `unable to reach health endpoint: ${e.message}` }
    }

    const enriched = { ...prediction, backend_health: health }
    const payload = JSON.stringify(enriched, null, 2)
    downloadBlob(payload, filename, "application/json")
  }

  const downloadReportTXT = async () => {
    if (!prediction) return
    const base = (prediction.original_jpeg || "report").replace(/\s+/g, "_")
    const filename = `${base}.txt`
    const lines = []
    lines.push(`Label: ${prediction.label}`)
    lines.push(`Confidence: ${prediction.confidence ?? "-"}`)
    lines.push(`Verdict: ${prediction.verdict ?? "-"}`)
    lines.push("")
    if (prediction.detection_stats) {
      lines.push("Detection stats:")
      Object.entries(prediction.detection_stats).forEach(([k, v]) => {
        lines.push(`  ${k}: ${v}`)
      })
      lines.push("")
    }
    lines.push("Defects:")
    if (Array.isArray(prediction.defects) && prediction.defects.length) {
      prediction.defects.forEach((d, i) => {
        lines.push(`${i + 1}. Component: ${d.component || "-"}; Type: ${d.type || "-"}; Confidence: ${d.confidence ?? "-"}`)
      })
    } else {
      lines.push("  (none)")
    }

    // append backend /health info
    lines.push("")
    lines.push("--- Backend health ---")
    try {
      const res = await fetch(`${API_BASE_URL}/health`)
      if (res.ok) {
        const health = await res.json()
        Object.entries(health).forEach(([k, v]) => {
          lines.push(`${k}: ${typeof v === 'object' ? JSON.stringify(v) : v}`)
        })
      } else {
        lines.push(`health endpoint returned ${res.status}`)
      }
    } catch (e) {
      lines.push(`unable to reach health endpoint: ${e.message}`)
    }

    downloadBlob(lines.join("\n"), filename, "text/plain")
  }

  const maskOverlayUrl = useMemo(() => {
    if (!prediction?.mask_overlay) {
      return null
    }
    return `${SEGMENTATION_BASE_URL}/${prediction.mask_overlay}`
  }, [prediction])

  const verdictBadgeClass = useMemo(() => {
    if (!prediction?.label) {
      return "badge"
    }
    return isPassingLabel(prediction.label) ? "badge positive" : "badge negative"
  }, [prediction])

  return (
    <div className="app-shell">
      <nav className="top-bar">
        <div className="brand">
          <div className="brand-stack">
            <img src={NeoDataLogo} alt="NeoData" className="brand-logo neo" />
          </div>
          <span className="brand-separator" aria-hidden="true" />
          <div className="brand-stack">
            <img src={KFK_LOGO_URL} alt="KFK" className="brand-logo kfk" />
            <small>KFK QC Challenge</small>
          </div>

        </div>
        <div className="top-actions">
          <div className="partner-logo">
            <img src={COMMINUS_LOGO_URL} alt="Comminus" />
          </div>
          <Link className="ghost secondary" to="/presentation">
            Presentation
          </Link>
          <a className="ghost secondary" href="https://www.kfk.hr" target="_blank" rel="noreferrer">
            KFK Portal
          </a>
        </div>
      </nav>

      <main className="content">
        <section className="hero">
          <div className="hero-text">
            <p className="eyebrow">Automatska kontrola kvalitete fasadnih elemenata</p>
            <h1>Učitaj fotografiju elementa i provjeri prolazi li strogu KFK inspekciju.</h1>
            <p className="subtitle">
              Sustav razlikuje <span className="highlight">POSITIVE</span> (ispravne) i{" "}
              <span className="highlight">NEGATIVE</span> (neispravne) uzorke te opisuje defekte poput nedostajućih
              vijaka, oštećenih brtvi, rupa ili puknutog stakla.
            </p>
            <ul className="hero-list">
              <li>Strukturirani izlaz: PASS/FAIL + popis detektiranih defekata.</li>
              <li>Uploadamo originalnu fotografiju (HEIC/JPG/PNG/BMP) bez dodatnog označavanja.</li>
              <li>API vraća JSON koji se može direktno ugraditi u QC izvještaj.</li>
            </ul>
          </div>
        </section>

        <section className="workspace">
          <div className="panel upload-panel">
            <div className="panel-header">
              <h2>1 · Učitaj fotografiju fasadnog elementa</h2>
              {file && (
                <button type="button" className="ghost" onClick={handleReset}>
                  Resetiraj
                </button>
              )}
            </div>


              <label className={`dropzone ${file ? "has-file" : ""}`} htmlFor="file-upload">
                {previewUrl ? (
                  <img src={previewUrl} alt="pregled" className="preview" />
                ) : (
                  <>
                    <span className="drop-icon" aria-hidden="true">
                      UPLOAD
                    </span>
                    <p>Povucite ili kliknite kako biste dodali fotografiju fasadnog panela</p>
                    <small>Podržani formati: JPG, PNG, BMP · preporučeno 4K</small>
                  </>
                )}
              </label>
              <div className="filename-warning">
                <span role="img" aria-label="warning" style={{marginRight: 6, fontSize: '1em'}}>⚠️</span>
                <span>
                  <strong>Napomena:</strong> Radi demonstracijskih namjera, naziv slike <u>mora biti identičan</u> kao u datasetu (npr. "IMG_5346 2.jpg").
                  Inače analiza neće biti moguća.
                </span>
              </div>
            <input id="file-upload" type="file" accept="image/*" onChange={handleFileChange} hidden />

            <div className="actions">
              <button type="button" className="primary" onClick={handleSubmit} disabled={status === "processing"}>
                {status === "processing" ? "Analiza..." : "Analiziraj fasadni element"}
              </button>
              {error && <span className="error-text">{error}</span>}
            </div>
          </div>

          <div className="panel results-panel">
            <div className="panel-header">
              <h2>2 · Rezultat modela</h2>
            </div>

            {status === "idle" && <p className="muted">Model čeka novu fotografiju.</p>}
            {status === "processing" && <p className="muted">Model obraduje sliku...</p>}
            {prediction && status === "done" && (
              <div className="result-card">
                <div className="result-headline">
                  <span className={verdictBadgeClass}>{prediction.label}</span>
                  {confidencePercent && <span className="confidence-chip">{confidencePercent}</span>}
                </div>
              
                

                <div className="result-visuals">
                  {maskOverlayUrl && (
                    <div className="analysis-visuals">
                      <figure className="analysis-card">
                        <span className="meta-label">Segmentirani overlay</span>
                        <img src={maskOverlayUrl} alt="Segmentirani overlay" />
                        <figcaption>{prediction.mask_overlay}</figcaption>
                      </figure>
                    </div>
                  )}
                  <div className="result-meta">
                    <div>
                      <span className="meta-label">Fotografija</span>
                      <strong>{prediction.original_jpeg || file?.name || "—"}</strong>
                    </div>
                    <div>
                      <span className="meta-label">CAD model</span>
                      <strong>{prediction.matched_model || "—"}</strong>
                    </div>
                    <div>
                      <span className="meta-label">Segmentacija</span>
                      <span>{prediction.mask_relative || prediction.mask_source || "—"}</span>
                    </div>
                  </div>
                </div>

                <div className="result-details">
                  <p className="verdict-text">{prediction.verdict}</p>

                  {prediction.detection_stats && (
                    <div className="stats-grid">
                      <div>
                        <span className="stat-label">Tin pokrivenost</span>
                        <strong>{formatPercent(prediction.detection_stats.tin_coverage)}</strong>
                      </div>
                      <div>
                        <span className="stat-label">Staklo</span>
                        <strong>{formatPercent(prediction.detection_stats.glass_coverage)}</strong>
                      </div>
                      <div>
                        <span className="stat-label">Brtva</span>
                        <strong>{formatPercent(prediction.detection_stats.seal_coverage)}</strong>
                      </div>
                      <div>
                        <span className="stat-label">Vijci</span>
                        <strong>{prediction.detection_stats.screw_count ?? 0}</strong>
                      </div>
                      <div>
                        <span className="stat-label">Rupe</span>
                        <strong>{prediction.detection_stats.hole_count ?? 0}</strong>
                      </div>
                    </div>
                  )}

                  {Array.isArray(prediction.defects) && prediction.defects.length > 0 ? (
                    <ul className="defect-list">
                      {prediction.defects.map((defect, index) => (
                        <li key={`${defect.component || "defect"}-${defect.type || index}-${index}`}>
                          <div className="defect-main">
                            <strong>{defect.component || "Nepoznata komponenta"}</strong>
                            <span className="defect-type">{defect.type || "Bez tipa"}</span>
                            {typeof defect.confidence === "number" && (
                              <span className="defect-confidence">{formatPercent(defect.confidence * 100)}</span>
                            )}
                          </div>
                          {Array.isArray(defect.zones_missing) && defect.zones_missing.length > 0 && (
                            <div className="zone-strip danger">
                              <span className="strip-label">Zone bez pokrova</span>
                              <div className="zone-pills">
                                {defect.zones_missing.map((zone) => (
                                  <span className="zone-chip danger" key={`missing-${zone}`}>
                                    {zone}
                                  </span>
                                ))}
                              </div>
                            </div>
                          )}
                          {Array.isArray(defect.zones_present) && defect.zones_present.length > 0 && (
                            <div className="zone-strip success">
                              <span className="strip-label">Zone s pokrovom</span>
                              <div className="zone-pills">
                                {defect.zones_present.map((zone) => (
                                  <span className="zone-chip success" key={`present-${zone}`}>
                                    {zone}
                                  </span>
                                ))}
                              </div>
                            </div>
                          )}
                        </li>
                      ))}
                    </ul>
                  ) : (
                    <p className="muted">Nema detektiranih defekata iznad zadane granice povjerenja.</p>
                  )}
                </div>
                <div className="download-actions">
                  <button type="button" className="ghost download-btn" onClick={downloadReportJSON}>
                    Preuzmi JSON
                  </button>
                  <button type="button" className="ghost download-btn" onClick={downloadReportTXT}>
                    Preuzmi TXT
                  </button>
                </div>
              </div>
            )}

            {history.length > 0 && (
              <div className="history">
                <h3>Posljednje procjene</h3>
                <ul>
                  {history.map((item) => (
                    <li key={item.timestamp}>
                      <span>{new Date(item.timestamp).toLocaleTimeString()}</span>
                      <span className="history-image">{item.image || "—"}</span>
                      <span className={isPassingLabel(item.label) ? "chip positive" : "chip negative"}>
                        {item.label}
                      </span>
                      <span>{(item.confidence * 100).toFixed(0)}%</span>
                    </li>
                  ))}
                </ul>
              </div>
            )}
          </div>
        </section>
      </main>
      <footer className="site-footer">
        <div className="footer-inner">Tim FERSADA</div>
      </footer>
    </div>
  )
}

export default App
