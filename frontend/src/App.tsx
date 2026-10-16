import { Routes, Route, Navigate } from 'react-router-dom'
import Layout from './components/layout/Layout'
import ErrorBoundary from './components/ui/ErrorBoundary'
import PlayPage from './pages/PlayPage'
import AnalyzePage from './pages/AnalyzePage'
import EditorPage from './pages/EditorPage'
import InputPage from './pages/InputPage'

export default function App() {
  return (
    <ErrorBoundary>
      <Routes>
        <Route element={<Layout />}>
          <Route path="/play" element={<PlayPage />} />
          <Route path="/analyze" element={<AnalyzePage />} />
          <Route path="/editor" element={<EditorPage />} />
          <Route path="/input" element={<InputPage />} />
          <Route path="/" element={<Navigate to="/play" replace />} />
        </Route>
      </Routes>
    </ErrorBoundary>
  )
}
