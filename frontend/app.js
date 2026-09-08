document.addEventListener('DOMContentLoaded', () => {
    const recordBtn = document.getElementById('recordBtn');
    const recordIcon = document.getElementById('recordIcon');
    const recordStatus = document.getElementById('recordStatus');
    const recordingTimer = document.getElementById('recordingTimer');
    const timerVal = document.getElementById('timerVal');
    
    const audioFileInput = document.getElementById('audioFileInput');
    const textInput = document.getElementById('textInput');
    const translateTextBtn = document.getElementById('translateTextBtn');
    
    const hindiOutput = document.getElementById('hindiOutput');
    const santaliOutput = document.getElementById('santaliOutput');
    const audioPlayer = document.getElementById('audioPlayer');
    
    const asrMs = document.getElementById('asrMs');
    const transMs = document.getElementById('transMs');
    const ttsMs = document.getElementById('ttsMs');
    const totalMs = document.getElementById('totalMs');
    const ramUsage = document.getElementById('ramUsage');
    const gpuUsage = document.getElementById('gpuUsage');

    // Dynamic API Base URL (supports direct origin or fallback to port 8000)
    const API_BASE = (window.location.port === '8000' || !window.location.port) 
        ? '' 
        : 'http://localhost:8000';

    let audioContext = null;
    let audioStream = null;
    let recorderNode = null;
    let pcmBuffers = [];
    let isRecording = false;
    let timerInterval = null;
    let startTime = 0;

    // --- Web Audio API 16kHz PCM WAV Encoder ---
    function encodePCM16Wav(samples, sampleRate = 16000) {
        const buffer = new ArrayBuffer(44 + samples.length * 2);
        const view = new DataView(buffer);

        const writeString = (offset, str) => {
            for (let i = 0; i < str.length; i++) {
                view.setUint8(offset + i, str.charCodeAt(i));
            }
        };

        writeString(0, 'RIFF');
        view.setUint32(4, 36 + samples.length * 2, true);
        writeString(8, 'WAVE');
        writeString(12, 'fmt ');
        view.setUint32(16, 16, true); // Subchunk1Size
        view.setUint16(20, 1, true);  // AudioFormat (1 = PCM)
        view.setUint16(22, 1, true);  // NumChannels (1 = Mono)
        view.setUint32(24, sampleRate, true);
        view.setUint32(28, sampleRate * 2, true); // ByteRate
        view.setUint16(32, 2, true);  // BlockAlign
        view.setUint16(34, 16, true); // BitsPerSample
        writeString(36, 'data');
        view.setUint32(40, samples.length * 2, true);

        let offset = 44;
        for (let i = 0; i < samples.length; i++, offset += 2) {
            const s = Math.max(-1, Math.min(1, samples[i]));
            view.setInt16(offset, s < 0 ? s * 0x8000 : s * 0x7FFF, true);
        }

        return new Blob([view], { type: 'audio/wav' });
    }

    // --- Microphone Recording Logic (MediaRecorder API) ---
    let mediaRecorder = null;
    let audioChunks = [];

    recordBtn.addEventListener('click', async () => {
        if (!isRecording) {
            try {
                audioStream = await navigator.mediaDevices.getUserMedia({ 
                    audio: {
                        channelCount: 1,
                        sampleRate: 16000,
                        echoCancellation: true,
                        noiseSuppression: true
                    } 
                });
                audioChunks = [];
                
                let options = {};
                if (MediaRecorder.isTypeSupported('audio/webm;codecs=opus')) {
                    options.mimeType = 'audio/webm;codecs=opus';
                } else if (MediaRecorder.isTypeSupported('audio/webm')) {
                    options.mimeType = 'audio/webm';
                } else if (MediaRecorder.isTypeSupported('audio/mp4')) {
                    options.mimeType = 'audio/mp4';
                }
                
                mediaRecorder = new MediaRecorder(audioStream, options);
                
                mediaRecorder.ondataavailable = (e) => {
                    if (e.data.size > 0) {
                        audioChunks.push(e.data);
                    }
                };

                mediaRecorder.onstop = async () => {
                    const rawBlob = new Blob(audioChunks, { type: mediaRecorder.mimeType || 'audio/webm' });
                    
                    try {
                        // Decode raw browser WebM/Opus payload using native browser AudioContext
                        const arrayBuffer = await rawBlob.arrayBuffer();
                        const tempCtx = new (window.AudioContext || window.webkitAudioContext)();
                        const audioBuffer = await tempCtx.decodeAudioData(arrayBuffer);
                        
                        // Resample & mix down to 16kHz Mono PCM via OfflineAudioContext
                        const targetSr = 16000;
                        const numFrames = Math.ceil(audioBuffer.duration * targetSr);
                        const offlineCtx = new OfflineAudioContext(1, numFrames, targetSr);
                        const source = offlineCtx.createBufferSource();
                        source.buffer = audioBuffer;
                        source.connect(offlineCtx.destination);
                        source.start(0);
                        
                        const renderedBuffer = await offlineCtx.startRendering();
                        const pcmSamples = renderedBuffer.getChannelData(0);
                        
                        // Encode to standard 16kHz 16-bit PCM WAV Blob
                        const wavBlob = encodePCM16Wav(pcmSamples, targetSr);
                        await processAudioPayload(wavBlob, 'recorded_speech.wav');
                        tempCtx.close();
                    } catch (err) {
                        // Fallback: send raw blob directly if Web Audio API decode fails
                        await processAudioPayload(rawBlob, 'recorded_speech.webm');
                    }
                };

                mediaRecorder.start(100);
                isRecording = true;
                
                // UI Updates
                recordBtn.classList.remove('bg-indigo-600', 'hover:bg-indigo-500');
                recordBtn.classList.add('bg-rose-600', 'hover:bg-rose-500', 'animate-pulse');
                recordIcon.className = 'fas fa-stop';
                recordStatus.textContent = 'Recording Hindi speech... Click to stop & process.';
                recordingTimer.classList.remove('hidden');
                
                startTime = Date.now();
                timerInterval = setInterval(updateTimer, 1000);
            } catch (err) {
                alert('Microphone access error: ' + err.message);
            }
        } else {
            // Stop recording
            isRecording = false;
            
            if (mediaRecorder && mediaRecorder.state !== 'inactive') {
                mediaRecorder.stop();
            }
            if (audioStream) {
                audioStream.getTracks().forEach(track => track.stop());
            }

            // Reset UI
            recordBtn.classList.remove('bg-rose-600', 'hover:bg-rose-500', 'animate-pulse');
            recordBtn.classList.add('bg-indigo-600', 'hover:bg-indigo-500');
            recordIcon.className = 'fas fa-microphone';
            recordStatus.textContent = 'Processing speech pipeline...';
            recordingTimer.classList.add('hidden');
            clearInterval(timerInterval);
        }
    });

    function updateTimer() {
        const elapsedSec = Math.floor((Date.now() - startTime) / 1000);
        const mins = String(Math.floor(elapsedSec / 60)).padStart(2, '0');
        const secs = String(elapsedSec % 60).padStart(2, '0');
        timerVal.textContent = `${mins}:${secs}`;
    }

    // --- Audio File Upload Event ---
    audioFileInput.addEventListener('change', async (e) => {
        const file = e.target.files[0];
        if (file) {
            recordStatus.textContent = `Processing file ${file.name}...`;
            await processAudioPayload(file, file.name);
        }
    });

    // --- Process Audio Payload via Backend API ---
    async function processAudioPayload(blobOrFile, filename) {
        setLoadingState(true);
        const formData = new FormData();
        formData.append('file', blobOrFile, filename);

        try {
            const response = await fetch(`${API_BASE}/api/translation-speech/hindi-to-santali`, {
                method: 'POST',
                body: formData
            });

            if (!response.ok) {
                const errData = await response.json();
                throw new Error(errData.detail || 'Audio translation pipeline failed.');
            }

            const data = await response.json();
            displayResults(data);
            recordStatus.textContent = 'Click to record Hindi speech';
        } catch (err) {
            alert('Pipeline Error: ' + err.message);
            recordStatus.textContent = 'Error processing audio. Try again.';
        } finally {
            setLoadingState(false);
        }
    }

    // --- Direct Text Translation Event ---
    translateTextBtn.addEventListener('click', async () => {
        const text = textInput.value.trim();
        if (!text) {
            alert('Please enter text to translate.');
            return;
        }

        const directionSelect = document.getElementById('translationDirection');
        const direction = directionSelect ? directionSelect.value : 'hin_Deva->sat_Olck';
        const [srcLang, tgtLang] = direction.split('->');

        hindiOutput.innerHTML = `<span class="text-white">${text}</span>`;
        santaliOutput.innerHTML = `<span class="text-slate-400 italic"><i class="fas fa-spinner fa-spin mr-2"></i>Translating...</span>`;
        transMs.textContent = '...';

        try {
            const endpoint = (srcLang === 'sat_Olck')
                ? `${API_BASE}/api/translation/santali-to-hindi`
                : `${API_BASE}/api/translation/hindi-to-santali`;

            const response = await fetch(endpoint, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    text: text,
                    source_language: srcLang,
                    target_language: tgtLang
                })
            });


            if (!response.ok) {
                const errData = await response.json();
                throw new Error(errData.detail || 'Text translation failed.');
            }

            const data = await response.json();
            santaliOutput.innerHTML = `<span class="text-indigo-200 font-semibold">${data.translated_text}</span>`;
            transMs.textContent = `${data.translation_ms} ms`;
            totalMs.textContent = `${data.translation_ms} ms`;
        } catch (err) {
            alert('Translation Error: ' + err.message);
            santaliOutput.innerHTML = `<span class="text-rose-400">Translation error: ${err.message}</span>`;
        }
    });

    // --- Display Pipeline Results & Latency Metrics ---
    function displayResults(data) {
        hindiOutput.innerHTML = `<span class="text-white font-medium">${data.hindi_text || '[Empty ASR]'}</span>`;
        santaliOutput.innerHTML = `<span class="text-indigo-200 font-semibold">${data.santali_text || '[Empty Translation]'}</span>`;

        if (data.audio_url) {
            audioPlayer.src = data.audio_url;
            audioPlayer.play().catch(() => {}); // Auto-play if permitted
        }

        if (data.latency_ms) {
            asrMs.textContent = `${data.latency_ms.asr_ms} ms`;
            transMs.textContent = `${data.latency_ms.translation_ms} ms`;
            ttsMs.textContent = `${data.latency_ms.tts_ms} ms`;
            totalMs.textContent = `${data.latency_ms.total_ms} ms`;
        }

        if (data.system_metrics) {
            if (data.system_metrics.ram_usage_mb) {
                ramUsage.textContent = `${data.system_metrics.ram_usage_mb} MB`;
            }
            if (data.system_metrics.gpu_memory_allocated_mb !== null) {
                gpuUsage.textContent = `${data.system_metrics.gpu_memory_allocated_mb} MB`;
            } else {
                gpuUsage.textContent = 'N/A (CPU)';
            }
        }
    }

    function setLoadingState(isLoading) {
        if (isLoading) {
            hindiOutput.innerHTML = '<span class="text-slate-400 animate-pulse"><i class="fas fa-spinner fa-spin mr-2"></i>Processing ASR (IndicConformer)...</span>';
            santaliOutput.innerHTML = '<span class="text-slate-400 animate-pulse"><i class="fas fa-spinner fa-spin mr-2"></i>Translating & Synthesizing (Parler-TTS)...</span>';
        }
    }
});
