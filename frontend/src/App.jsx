import { useState } from 'react'
import AudioEmotionAnalyzer from "./components/AudioEmotionAnalyzer" 
import './App.css'

function App() {
  const [count, setCount] = useState(0)

  return (
    <>
      <AudioEmotionAnalyzer/>
    </>
  )
}

export default App
