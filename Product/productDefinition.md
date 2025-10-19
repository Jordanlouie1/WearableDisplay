# Product Definition Report: ComeFriend – Intelligent Gesture-Activated AI Camera

## 1. Product Overview

**Concept**  
WearableDisplay (code name: ComeFriend) is an intelligent camera experience that lets users control capture, cropping, and sharing through hand gestures and audio cues—no screen touch required. Captured or cropped images are routed to a multimodal LLM chatbot for contextual understanding, analysis, and dialogue.

**Vision Statement**  
Deliver a hands-free companion that understands visual intent and provides actionable conversational feedback.

**Value Proposition**  
- Enables touchless image workflows for accessibility and productivity.  
- Bridges real-time capture with multimodal AI reasoning.  
- Reduces friction in environments where manual interaction is impractical.

## 2. Objectives

- Build an intuitive gesture-first interface for capture, edit, and confirmation.  
- Supplement gestures with audio triggers (voice, clap) to increase accessibility. Trigger is done by circling the image when the hand (scope) or by saying "ComeFriend go"
- Integrate a seamless pipeline from capture → crop → encode → LLM → response.  
- Maintain modularity to port from desktop MVP to mobile platforms.

## 3. Key Features

| Feature | Description | Technology / Tools |
| --- | --- | --- |
| Gesture Activation | Recognizes palm-open activation, pinch-to-frame, thumbs-up confirmation, with confidence scoring and fallback prompts. | Mediapipe Hands/Holistic, OpenCV, TensorFlow Lite |
| Sound Activation | Listens for wake word, double clap, or short voice commands to trigger capture, retake, or send. | PyAudio, Whisper (tiny/medium) or Vosk, FFT clap detector |
| Gesture-Based Cropping | Generates live bounding boxes from fingertip landmarks, applies smoothing, and supports auto-snap to detected objects. | Mediapipe landmarks, Kalman filter, OpenCV image ops |
| LLM Integration | Encodes captured frames, sends to multimodal LLM, streams conversational response with history awareness. | LangChain, Openrouter API (e.g., GPT-4V, Claude 3 Opus Vision) |
| Cross-Platform Delivery | Ships desktop prototype and mobile apps sharing a core inference layer. | React Native or Flutter UI, Electron/Flutter desktop shell |

## 4. System Architecture

**Input Layer**  
- Camera feed (front or rear, 1080p recommended).
- Microphone stream (16 kHz), optional array for noise reduction.

**Processing Layer**  
- Frame pre-processing (resolution scaling, color normalization).  
- Gesture detection graph: Mediapipe → classifier → confidence thresholds.  
- Audio detection: streaming FFT, clap detector, Whisper-based ASR with wake word filter.  
- State manager to debounce and synchronize gesture/audio events.  
- Capture and crop module: freeze frame, apply bounding box, allow manual override.

**LLM Interaction Layer**  
- Image encoder (JPEG/WebP) with size heuristics and optional CLIP embeddings.  
- LangChain orchestrator forming prompts with context and history.  
- Multimodal API call via Openrouter to GPT-4 Vision or Claude.  
- Response handler streaming text, optional JSON schema for structured replies.

**UI / UX Layer**  
- Real-time camera preview with landmark overlay and crop ghost.  
- Feedback cues: color change, sound, optional haptics.  
- Chat interface displaying user commands and AI responses.  
- Accessibility controls: high-contrast mode, audible narration, gesture tutorial.

## 5. Technical Requirements

**Hardware**  
- Device with camera, microphone, and mid-range CPU/GPU/NPUs (mobile or laptop).  
- Stable power and thermal management for sustained inference.

**Software Stack**  
- Frontend: React Native or Flutter for mobile; Electron/Flutter for desktop MVP.  
- Backend: FastAPI (Python) or Node.js for CV services, REST/WebSocket APIs.  
- Vision: Mediapipe, OpenCV, TensorFlow Lite (quantized models for mobile).  
- Audio: PyAudio, Whisper API or Whisper.cpp for offline mode.  
- LLM Access: Openrouter API key management, rate limiting, retries.  
- DevOps: Dockerized services, GitHub Actions CI, DVC for model versioning.

## 6. User Scenarios

- **Teacher** waves to activate, frames an object, says “Analyze this object,” and receives a descriptive explanation for lesson planning.  
- **Home cook** claps twice, captures the countertop, asks “What ingredients do I still need?” and gets guidance against a recipe list.  
- **Designer** sketches in the air to crop a whiteboard section, sends it to the chatbot, and receives ideation prompts and refinements.

## 7. Roadmap

| Phase | Milestone | Timeline | Deliverables |
| --- | --- | --- | --- |
| Phase 1 | Desktop MVP with gesture + sound detection | Day-1 | Working prototype showcasing capture and cropping locally |
| Phase 2 | LLM integration and chat loop | Day-1 | Multimodal conversation demo with Openrouter endpoint |
| Phase 3 | Mobile app beta | Weeks 1-6 | React Native or Flutter build with optimized models |
| Phase 4 | UX polish and cloud sync | Weeks 6-10 | Launch-ready app with user accounts, optional sync, analytics |

## 8. Potential Applications

- Accessibility assistant for mobility-limited users requiring touchless interaction.  
- Education companion that responds to visual prompts in classrooms.  
- Design and creativity aide for rapid ideation using sketches and prototypes.  
- Retail associate tool enabling gesture-based product lookup or styling advice.

## 9. Risks and Mitigations

| Risk | Impact | Mitigation |
| --- | --- | --- |
| Gesture misdetection | Frustrating UX, missed actions | Multi-stage classifier, per-user calibration, fallback UI buttons |
| LLM latency | Sluggish responses, user drop-off | Async calls, progress indicators, quick summary while full reply streams |
| Privacy concerns | User trust erosion | On-device preprocessing, explicit consent, anonymize stored data |
| Platform constraints | Feature gaps on low-end devices | Modular services, lightweight/quantized models, offline cache |
| Battery & thermal load | Mobile usability degradation | Adaptive frame rate, hardware acceleration, auto-idle mode |

## 10. Success Metrics

- ≥90% gesture recognition accuracy in controlled usability tests.  
- <2 seconds median latency from gesture to action on desktop MVP.  
- ≥80% conversational task completion rate during pilot.  
- Net Promoter Score ≥75 from accessibility and education cohorts.

## 11. Next Steps

1. Develop desktop MVP integrating Mediapipe gesture capture, clap detection, and stubbed LLM request.  
2. Conduct usability tests to refine gesture vocabulary, feedback cues, and error recovery.  
3. Finalize Openrouter integration plan (key management, rate limits, error handling).  
4. Outline data privacy policy and consent flow ahead of beta release.
