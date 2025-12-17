import os
import cv2
import av
import numpy as np
import pickle
import streamlit as st
import mediapipe as mp
from typing import Optional, Tuple
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Try importing TensorFlow with detailed error handling
try:
    import tensorflow as tf
    # Suppress TensorFlow warnings
    tf.get_logger().setLevel('ERROR')
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
except ImportError as e:
    st.error("❌ TensorFlow is not installed. Please check requirements.txt.")
    st.info("Run: `pip install tensorflow`")
    st.stop()
except Exception as e:
    st.error(f"Error importing TensorFlow: {e}")
    st.stop()

try:
    from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration, WebRtcMode
except ImportError:
    st.error("❌ streamlit-webrtc is not installed.")
    st.info("Run: `pip install streamlit-webrtc`")
    st.stop()

# ==========================================
# CONFIGURATION
# ==========================================
st.set_page_config(
    page_title="AI Sign Language Translator",
    page_icon="",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Force CPU usage to prevent GPU memory errors
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# Constants
STABILITY_FRAMES = 15
CONFIDENCE_THRESHOLD = 0.85
MODEL_PATH = "sign_language_model.keras"
ENCODER_PATH = "label_encoder.pkl"
MAX_SENTENCE_LENGTH = 100

# STUN Server Configuration
RTC_CONFIGURATION = RTCConfiguration({
    "iceServers": [
        {"urls": ["stun:stun.l.google.com:19302"]},
        {"urls": ["stun:stun1.l.google.com:19302"]}
    ]
})

# ==========================================
# RESOURCE LOADING (CACHED & SAFE)
# ==========================================
@st.cache_resource
def load_resources() -> Optional[Tuple]:
    """
    Load Model, Encoder, and MediaPipe safely.
    Returns: (model, encoder, hands, mp_hands, mp_drawing) or None
    """
    try:
        # Check file existence
        if not os.path.exists(MODEL_PATH):
            logger.error(f"Model file not found: {MODEL_PATH}")
            return None
        
        if not os.path.exists(ENCODER_PATH):
            logger.error(f"Encoder file not found: {ENCODER_PATH}")
            return None

        # Load Keras Model with error handling
        try:
            model = tf.keras.models.load_model(MODEL_PATH, compile=False)
            logger.info("Model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return None
        
        # Load Label Encoder
        try:
            with open(ENCODER_PATH, "rb") as f:
                encoder = pickle.load(f)
            logger.info("Encoder loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load encoder: {e}")
            return None

        # Setup MediaPipe with error handling
        try:
            mp_hands = mp.solutions.hands
            mp_drawing = mp.solutions.drawing_utils
            hands = mp_hands.Hands(
                static_image_mode=False,
                max_num_hands=1,
                min_detection_confidence=0.7,
                min_tracking_confidence=0.5
            )
            logger.info("MediaPipe initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize MediaPipe: {e}")
            return None

        return model, encoder, hands, mp_hands, mp_drawing
    
    except Exception as e:
        logger.error(f"Unexpected error loading resources: {e}")
        return None

# ==========================================
# VIDEO PROCESSOR CLASS
# ==========================================
class SignLanguageProcessor(VideoProcessorBase):
    def __init__(self):
        self.sentence = []
        self.last_pred = None
        self.frame_counter = 0
        self.error_count = 0
        self.max_errors = 10
        
    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        """
        Process each video frame safely with comprehensive error handling.
        """
        try:
            # Convert frame to numpy array
            img = frame.to_ndarray(format="bgr24")
            
            # Validate frame
            if img is None or img.size == 0:
                logger.warning("Invalid frame received")
                return frame
            
            # Pre-processing
            image = cv2.flip(img, 1)  # Mirror effect
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Initialize default values
            predicted_char = "..."
            hand_detected = False
            
            # Hand Detection with error handling
            try:
                results = hands.process(rgb)
            except Exception as e:
                logger.error(f"MediaPipe processing error: {e}")
                self.error_count += 1
                if self.error_count > self.max_errors:
                    logger.critical("Too many errors, resetting")
                    self.error_count = 0
                return av.VideoFrame.from_ndarray(image, format="bgr24")

            # Process hand landmarks
            if results and results.multi_hand_landmarks:
                hand_detected = True
                
                try:
                    for hand_landmarks in results.multi_hand_landmarks:
                        # Draw landmarks
                        mp_drawing.draw_landmarks(
                            image, 
                            hand_landmarks, 
                            mp_hands.HAND_CONNECTIONS,
                            mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=4),
                            mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2)
                        )

                        # Extract landmarks
                        landmarks = []
                        for lm in hand_landmarks.landmark:
                            landmarks.extend([lm.x, lm.y])

                        # Validate landmark data
                        if len(landmarks) != 42:  # 21 landmarks * 2 coordinates
                            logger.warning(f"Invalid landmark count: {len(landmarks)}")
                            continue

                        # AI Prediction with error handling
                        try:
                            input_data = np.array([landmarks], dtype=np.float32)
                            
                            # Validate input shape
                            if input_data.shape[1] != 42:
                                logger.warning(f"Invalid input shape: {input_data.shape}")
                                continue
                            
                            prediction = model.predict(input_data, verbose=0)
                            class_id = np.argmax(prediction)
                            confidence = np.max(prediction)

                            if confidence > CONFIDENCE_THRESHOLD:
                                current_char = str(encoder.inverse_transform([class_id])[0])
                                
                                # Stabilization Logic
                                if current_char == self.last_pred:
                                    self.frame_counter += 1
                                else:
                                    self.frame_counter = 0
                                    self.last_pred = current_char

                                if self.frame_counter >= STABILITY_FRAMES:
                                    self._update_sentence(current_char)
                                    self.frame_counter = 0
                                    predicted_char = current_char
                                else:
                                    predicted_char = self.last_pred if self.last_pred else "..."
                            
                            # Reset error count on success
                            self.error_count = 0
                            
                        except Exception as e:
                            logger.error(f"Prediction error: {e}")
                            self.error_count += 1
                            
                except Exception as e:
                    logger.error(f"Landmark processing error: {e}")
                    self.error_count += 1

            # Draw UI overlay
            self._draw_overlay(image, predicted_char, hand_detected)
            
            return av.VideoFrame.from_ndarray(image, format="bgr24")
            
        except Exception as e:
            logger.error(f"Frame processing error: {e}")
            self.error_count += 1
            # Return original frame if processing fails
            try:
                return av.VideoFrame.from_ndarray(frame.to_ndarray(format="bgr24"), format="bgr24")
            except:
                return frame
    
    def _update_sentence(self, char: str):
        """Update sentence with character, handling special commands."""
        try:
            char_lower = char.lower()
            
            if "space" in char_lower:
                self.sentence.append(" ")
            elif "delete" in char_lower or "backspace" in char_lower:
                if self.sentence:
                    self.sentence.pop()
            else:
                self.sentence.append(char)
            
            # Limit sentence length to prevent memory issues
            if len(self.sentence) > MAX_SENTENCE_LENGTH:
                self.sentence = self.sentence[-MAX_SENTENCE_LENGTH:]
                
        except Exception as e:
            logger.error(f"Sentence update error: {e}")
    
    def _draw_overlay(self, image: np.ndarray, predicted_char: str, hand_detected: bool):
        """Draw UI overlay on the image."""
        try:
            h, w, _ = image.shape
            
            # Top Bar Background
            cv2.rectangle(image, (0, 0), (w, 100), (20, 20, 20), -1)
            
            # Status Indicator
            status_color = (0, 255, 0) if hand_detected else (0, 0, 255)
            cv2.circle(image, (w - 30, 30), 10, status_color, -1)

            # Prediction Text
            cv2.putText(
                image, "DETECTING:", (20, 35), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1, cv2.LINE_AA
            )
            cv2.putText(
                image, f"{predicted_char}", (150, 38), 
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2, cv2.LINE_AA
            )

            # Sentence Display
            sentence_str = "".join(self.sentence)[-30:]  # Last 30 chars
            cv2.putText(
                image, "SENTENCE:", (20, 80), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1, cv2.LINE_AA
            )
            cv2.putText(
                image, sentence_str, (150, 83), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA
            )
            
        except Exception as e:
            logger.error(f"Overlay drawing error: {e}")

# ==========================================
# MAIN UI
# ==========================================
def main():
    st.title("AI Sign Language Translator (ASL)")
    st.markdown("### Real-time Hand Gesture Recognition")
    
    # Load Resources
    resources = load_resources()
    
    if resources is None:
        st.error("❌ Failed to load required resources!")
        st.warning(f"""
        **Deployment Checklist:**
        
        Ensure these files exist in your deployment directory:
        1. ✅ `{MODEL_PATH}` - TensorFlow model file
        2. ✅ `{ENCODER_PATH}` - Label encoder file
        
        **Troubleshooting:**
        - Verify files are uploaded to your Streamlit Cloud repository
        - Check file names match exactly (case-sensitive)
        - Ensure files are in the root directory or update paths
        """)
        
        # Show current directory contents for debugging
        with st.expander("📁 Current Directory Contents"):
            try:
                files = os.listdir(".")
                st.code("\n".join(files))
            except Exception as e:
                st.error(f"Cannot list directory: {e}")
        
        st.stop()
    
    # Unpack resources globally
    global model, encoder, hands, mp_hands, mp_drawing
    model, encoder, hands, mp_hands, mp_drawing = resources
    
    st.success("✅ All resources loaded successfully!")
    
    # Instructions
    st.markdown("### 📷 Camera Feed")
    st.info("""
    **Instructions:**
    - Position your hand clearly in front of the camera
    - Ensure good lighting
    - Hold each sign steady for recognition
    - Use "Space" and "Delete" gestures for sentence building
    """)
    
    # WebRTC Streamer
    try:
        ctx = webrtc_streamer(
            key="sign-language-translator",
            mode=WebRtcMode.SENDRECV,
            rtc_configuration=RTC_CONFIGURATION,
            video_processor_factory=SignLanguageProcessor,
            media_stream_constraints={"video": True, "audio": False},
            async_processing=True,
        )
        
        # Display status
        if ctx.state.playing:
            st.success("🟢 Camera is active")
        else:
            st.info("⚪ Click START to begin")
            
    except Exception as e:
        st.error(f"Camera initialization error: {e}")
        st.info("""
        **Troubleshooting Camera Issues:**
        - Grant camera permissions in your browser
        - Try a different browser (Chrome/Edge recommended)
        - Check if another application is using the camera
        - Refresh the page and try again
        """)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: gray;'>
        Made with using Streamlit, TensorFlow & MediaPipe
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()