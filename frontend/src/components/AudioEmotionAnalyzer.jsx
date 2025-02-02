import React, { useState, useRef, useEffect } from 'react';
import axios from 'axios';
import { Mic, Play, Square, Upload } from 'lucide-react';

const AudioEmotionAnalyzer = () => {
  const [isRecording, setIsRecording] = useState(false);
  const [audioFile, setAudioFile] = useState(null);
  const [emotionData, setEmotionData] = useState([]);
  const [isPlaying, setIsPlaying] = useState(false);
  const [currentEmotion, setCurrentEmotion] = useState('');
  const [error, setError] = useState(null);
  const [dragActive, setDragActive] = useState(false);

  const audioRef = useRef(null);
  const mediaRecorderRef = useRef(null);
  const chunksRef = useRef([]);
  const canvasRef = useRef(null);

  useEffect(() => {
    if (audioFile && canvasRef.current) {
      visualizeAudio();
    }
  }, [audioFile]);

  useEffect(() => {
    let animationFrameId;

    const syncEmotionWithAudio = () => {
      if (isPlaying && audioRef.current && emotionData.length > 0) {
        const currentTime = audioRef.current.currentTime;

        const matchedEmotion = emotionData.find(
          ({ Start_time, End_time }) => currentTime >= Start_time && currentTime < End_time
        );
        setCurrentEmotion(matchedEmotion ? matchedEmotion.Prediction : '');

        animationFrameId = requestAnimationFrame(syncEmotionWithAudio);
      }
    };

    if (isPlaying) {
      animationFrameId = requestAnimationFrame(syncEmotionWithAudio);
    }

    return () => cancelAnimationFrame(animationFrameId);
  }, [isPlaying, emotionData]);

  const visualizeAudio = async () => {
    if (!audioRef.current) return;

    const audioContext = new (window.AudioContext || window.webkitAudioContext)();
    const analyser = audioContext.createAnalyser();
    let source;

    // Ensure the previous source is disconnected
    try {
      source = audioContext.createMediaElementSource(audioRef.current);
    } catch (err) {
      // console.log("Audio already connected; skipping reconnection.");
      return;
    }

    source.connect(analyser);
    analyser.connect(audioContext.destination);

    analyser.fftSize = 256;
    const bufferLength = analyser.frequencyBinCount;
    const dataArray = new Uint8Array(bufferLength);

    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    const WIDTH = canvas.width;
    const HEIGHT = canvas.height;

    const draw = () => {
      requestAnimationFrame(draw);
      analyser.getByteFrequencyData(dataArray);

      ctx.fillStyle = '#1f2937';
      ctx.fillRect(0, 0, WIDTH, HEIGHT);

      const barWidth = (WIDTH / bufferLength) * 2.5;
      let barHeight;
      let x = 0;

      for (let i = 0; i < bufferLength; i++) {
        barHeight = dataArray[i] / 2;
        ctx.fillStyle = getEmotionColor();
        ctx.fillRect(x, HEIGHT - barHeight, barWidth, barHeight);
        x += barWidth + 1;
      }
    };

    draw();
  };

  const getEmotionColor = () => {
    const colors = {
      sadness: '#60a5fa',
      disgust: '#c084fc',
      anger: '#f87171',
      happiness: '#4ade80',
      fear: '#facc15',
      surprise: '#fbbf24',
      neutral: '#9ca3af'
    };
    return colors[currentEmotion] || '#9ca3af';
  };

  const handleFileUpload = (file) => {
    if (file) {
      setAudioFile(URL.createObjectURL(file));

      if (audioRef.current) {
        audioRef.current.pause();
        audioRef.current.currentTime = 0;
      }
      sendAudioToBackend(file);
    }
  };


  const sendAudioToBackend = async (file) => {
    try {
      const formData = new FormData();
      formData.append('audio_file', file);

      const response = await axios.post('http://127.0.0.1:5003/audio/emotions', formData, {
        headers: {
          'Content-Type': 'multipart/form-data' // Axios should automatically set this for you
        }
      })
      setEmotionData(response.data);
      setError(null);
    } catch (err) {
      setError('Failed to process audio. Please try again.');
    }
  };

  const togglePlayback = () => {
    if (audioRef.current) {
      if (isPlaying) {
        audioRef.current.pause();
      } else {
        audioRef.current.play();
      }
      setIsPlaying(!isPlaying);
    }
  };

  return (
    <div className="max-w-2xl mx-auto p-6 bg-gray-900 rounded-2xl shadow-xl mt-5">
      {/* Upload Area */}

      <div
        className={`relative p-8 mb-6 border-2 border-dashed rounded-xl transition-colors
          ${dragActive ? 'border-blue-500 bg-blue-50/5' : 'border-gray-600 hover:border-gray-500'}`}
        onDragEnter={(e) => {
          e.preventDefault();
          setDragActive(true);
        }}
        onDragLeave={(e) => {
          e.preventDefault();
          setDragActive(false);
        }}
        onDragOver={(e) => {
          e.preventDefault();
          setDragActive(true);
        }}
        onDrop={(e) => {
          e.preventDefault();
          setDragActive(false);
          const file = e.dataTransfer.files[0];
          if (file) handleFileUpload(file);
        }}
      >
        {
          audioFile ? "" : ""
        }
        <input
          type="file"
          accept="audio/*"
          onChange={(e) => handleFileUpload(e.target.files[0])}
          className="absolute inset-0 w-full h-full opacity-0 cursor-pointer"
        />
        <div className="flex flex-col items-center text-gray-400">
          {audioFile ? (
            <p className="text-lg font-medium text-green-400 mb-2">
              Uploaded File: {audioFile.split('/').pop()}
            </p>
          ) : (
            <>
              <Upload className="w-12 h-12 mb-4" />
              <p className="text-lg font-medium">Drop your audio file here or click to upload</p>
              <p className="text-sm">Supports WAV, MP3, M4A</p>
            </>
          )}
        </div>
      </div>

      {/* Controls */}
      <div className="flex gap-4 mb-6">
        {audioFile && (
          <button
            onClick={togglePlayback}
            className="flex items-center gap-2 px-6 py-3 bg-gray-700 hover:bg-gray-600 text-white rounded-lg font-medium transition-colors"
          >
            {isPlaying ? (
              <>
                <Square className="w-5 h-5" />
                Pause
              </>
            ) : (
              <>
                <Play className="w-5 h-5" />
                Play
              </>
            )}
          </button>
        )}
      </div>

      {/* Visualization */}
      {audioFile && (
        <div className="mb-6">
          <canvas
            ref={canvasRef}
            width="640"
            height="100"
            className="w-full rounded-lg bg-gray-800"
          />
          <audio
            ref={audioRef}
            src={audioFile}
            onEnded={() => setIsPlaying(false)}
            className="hidden"
          />
        </div>
      )}

      {/* Emotion Timeline */}
      {/* Emotion Timeline */}
      {emotionData.length > 0 && (
        <div className="mt-4 space-y-2">
          <h3 className="text-xl font-semibold text-white mb-2">Emotion Timeline</h3>
          <div className=' grid md:grid-cols-2 gap-4 p-3'>
            {emotionData.map(({ Prediction, Start_time, End_time }, index) => {
              const isActive = audioRef.current?.currentTime >= Start_time && audioRef.current?.currentTime < End_time;

              return (
                <div
                  key={index}
                  className={`w-64 flex justify-between items-center p-3 rounded-lg transition-colors ${isActive ? 'bg-green-700 text-white' : 'bg-gray-800 text-gray-300'
                    }`}
                >
                  <p className="capitalize font-medium">{Prediction}</p>
                  <p className="text-sm">{`Time: ${Start_time}s - ${End_time}s`}</p>
                </div>
              );
            })}
          </div>

        </div>
      )}

      {/* Error Display */}
      {error && (
        <div className="p-4 mb-6 rounded-lg bg-red-500 bg-opacity-10 text-red-400">
          <p className="font-medium">{error}</p>
        </div>
      )}
    </div>
  );
};

export default AudioEmotionAnalyzer;