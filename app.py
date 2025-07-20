import streamlit as st
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import cv2
from ultralytics import YOLO
from fpdf import FPDF
import tempfile
import os
import time
import pandas as pd
import subprocess
import imageio_ffmpeg
import uuid
import shutil

# ---- SET PAGE CONFIG FIRST! ----
st.set_page_config(page_title="RoadBot", layout="centered")

# ---- LOAD CUSTOM YOLOv8 MODEL ----
@st.cache_resource
def load_model():
    model = YOLO("best.pt")
    return model

model = load_model()

# ---- CUSTOM ANNOTATION FUNCTION FOR BETTER VISIBILITY ----
def draw_custom_annotations(frame, results, class_names):
    """
    Draw custom annotations with high contrast colors for better visibility
    """
    annotated_frame = frame.copy()
    
    if results.boxes is not None:
        for box in results.boxes:
            # Get box coordinates
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            confidence = float(box.conf[0])
            class_id = int(box.cls[0])
            class_name = class_names[class_id]
            
            # Choose high contrast colors
            colors = {
                'pothole': (0, 0, 255),      # Red
                'crack': (0, 255, 0),        # Green  
                'patch': (255, 0, 0),        # Blue
                'default': (0, 255, 255)     # Yellow
            }
            
            color = colors.get(class_name.lower(), colors['default'])
            
            # Draw bounding box with thick border
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 3)
            
            # Create label with confidence
            label = f"{class_name}: {confidence:.2f}"
            
            # Get text size for background rectangle
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.7
            thickness = 2
            (text_width, text_height), baseline = cv2.getTextSize(label, font, font_scale, thickness)
            
            # Draw background rectangle for text (semi-transparent)
            overlay = annotated_frame.copy()
            cv2.rectangle(overlay, (x1, y1 - text_height - 10), 
                         (x1 + text_width, y1), color, -1)
            cv2.addWeighted(overlay, 0.7, annotated_frame, 0.3, 0, annotated_frame)
            
            # Draw text in contrasting color
            text_color = (255, 255, 255) if sum(color) < 400 else (0, 0, 0)
            cv2.putText(annotated_frame, label, (x1, y1 - 5), 
                       font, font_scale, text_color, thickness)
    
    return annotated_frame

# ---- ENHANCED IMAGE DETECTION ----
def detect_image_damage(model, image, confidence_threshold=0.5):
    """
    Enhanced image detection with custom high-contrast visualization
    """
    try:
        # Convert PIL to OpenCV format
        cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        # Run inference
        results = model(cv_image, conf=confidence_threshold)
        
        # Try custom annotations first
        try:
            annotated_img = draw_custom_annotations(cv_image, results[0], model.names)
        except Exception as e:
            st.warning(f"Custom annotation failed, using default: {str(e)}")
            # Fallback to default YOLO plotting
            annotated_img = results[0].plot(
                conf=True,
                labels=True,
                boxes=True,
                line_width=3,
                font_size=14
            )
        
        # Convert back to PIL format
        return Image.fromarray(cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)), results
        
    except Exception as e:
        st.error(f"Error in image detection: {str(e)}")
        return image, None

# ---- FFmpeg RE-ENCODING (FROM CODE2) ----
def reencode_video_with_ffmpeg(input_path, output_path):
    """
    Re-encode video using FFmpeg for better web compatibility
    """
    try:
        ffmpeg_path = imageio_ffmpeg.get_ffmpeg_exe()
        cmd = [
            ffmpeg_path, '-y',
            '-i', input_path,
            '-vcodec', 'libx264',
            '-preset', 'fast',
            '-movflags', 'faststart',
            '-acodec', 'aac',
            '-strict', 'experimental',
            output_path
        ]
        subprocess.run(cmd, check=True, capture_output=True)
        return output_path
    except Exception as e:
        st.warning(f"FFmpeg re-encoding failed: {e}")
        return None

# ---- IMPROVED VIDEO PROCESSING WITH DIRECT PLAYBACK ----
def detect_video_damage_with_playback(model, video_file, confidence_threshold=0.5):
    """
    Process video with YOLO detection and create web-compatible output for direct playback
    """
    # Save uploaded video temporarily
    temp_input = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    temp_input.write(video_file.read())
    temp_input.close()
    
    cap = cv2.VideoCapture(temp_input.name)
    
    if not cap.isOpened():
        st.error("Error: Could not open video file")
        return None, None, {}
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS) or 20
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) & ~1  # Ensure even width
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) & ~1  # Ensure even height
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Create raw output path
    raw_output = os.path.join(tempfile.gettempdir(), f"raw_output_{uuid.uuid4().hex[:6]}.mp4")
    
    # Use mp4v codec for initial processing
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(raw_output, fourcc, fps, (width, height))
    
    if not out.isOpened():
        st.error("Error: Could not create output video writer")
        cap.release()
        return None, None, {}
    
    # For tracking detections across video
    detection_stats = {}
    frame_count = 0
    
    # Progress bar
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            frame_count += 1
            
            # Resize frame to ensure consistent dimensions
            frame = cv2.resize(frame, (width, height))
            
            # Run YOLO inference
            results = model(frame, conf=confidence_threshold)
            
            # Custom annotation for better visibility
            annotated_frame = draw_custom_annotations(frame, results[0], model.names)
            
            # Count detections in this frame
            if results[0].boxes is not None:
                for box in results[0].boxes:
                    class_id = int(box.cls[0])
                    confidence = float(box.conf[0])
                    class_name = model.names[class_id]
                    
                    if class_name not in detection_stats:
                        detection_stats[class_name] = []
                    detection_stats[class_name].append(confidence)
            
            # Write the annotated frame
            out.write(annotated_frame)
            
            # Update progress
            if total_frames > 0:
                progress = min(frame_count / total_frames, 1.0)
                progress_bar.progress(progress)
                status_text.text(f"Processing frame {frame_count}/{total_frames}")
    
    except Exception as e:
        st.error(f"Error during video processing: {str(e)}")
        return None, None, {}
    
    finally:
        cap.release()
        out.release()
        progress_bar.empty()
        status_text.empty()
    
    # Wait for file to be written
    time.sleep(1)
    
    # Re-encode for web compatibility using FFmpeg
    final_output = os.path.join(tempfile.gettempdir(), f"final_output_{uuid.uuid4().hex[:6]}.mp4")
    reencoded_path = reencode_video_with_ffmpeg(raw_output, final_output)
    
    if reencoded_path and os.path.exists(reencoded_path):
        # Wait for re-encoding to complete
        time.sleep(1)
        
        # Read video bytes for direct playback
        try:
            with open(reencoded_path, "rb") as f:
                video_bytes = f.read()
        except Exception as e:
            st.error(f"Error reading processed video: {str(e)}")
            return None, None, {}
        
        # Clean up temporary files
        try:
            os.unlink(temp_input.name)
            os.unlink(raw_output)
        except:
            pass
            
        return video_bytes, reencoded_path, detection_stats
    else:
        st.error("❌ Video processing failed. Please try again with a different video file.")
        return None, None, {}

# ---- ENHANCED REPORT GENERATION ----
def generate_enhanced_pdf_report(user_info, detected_img, damage_summary, confidence_stats=None):
    """
    Generate PDF report with enhanced formatting and confidence statistics
    """
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=20)
    pdf.add_page()

    # Add Logo (if exists)
    logo_path = r"C:\Users\Msi\Desktop\SEM6\FYP\yolo\prototype\logo.png"
    if os.path.exists(logo_path):
        try:
            pdf.image(logo_path, x=15, y=15, w=30)
        except Exception as e:
            st.warning(f"Could not load logo: {str(e)}")

    # Title
    pdf.set_font("Helvetica", "B", 18)
    pdf.cell(0, 25, "        Road Damage Detection Report", ln=True, align="C")

    # Slogan
    pdf.set_font("Helvetica", size=10)
    pdf.cell(0, 8, "Detect Fast. Repair Faster.", ln=True, align="C")
    pdf.ln(10)

    # User Info
    pdf.set_font("Helvetica", size=12)
    pdf.cell(0, 8, f"Inspector: {user_info['name']}", ln=True)
    pdf.cell(0, 8, f"Email: {user_info['email']}", ln=True)
    pdf.cell(0, 8, f"Location: {user_info['location']}", ln=True)
    pdf.cell(0, 8, f"Date: {time.strftime('%Y-%m-%d')}", ln=True)
    pdf.ln(10)

    # Add image if provided
    if detected_img:
        with tempfile.TemporaryDirectory() as tmpdir:
            detected_path = os.path.join(tmpdir, "detected.png")
            detected_img.save(detected_path)
            try:
                pdf.image(detected_path, x=15, w=75)
            except Exception as e:
                pdf.cell(0, 8, f"[Image could not be embedded: {str(e)}]", ln=True)
        pdf.ln(10)

    # Enhanced Summary Header
    pdf.set_font("Helvetica", "B", 14)
    pdf.cell(0, 10, "Detection Summary", ln=True)
    pdf.ln(5)

    # Enhanced Summary Table
    pdf.set_font("Helvetica", "B", 12)
    pdf.set_fill_color(230, 230, 230)
    pdf.cell(70, 8, "Damage Type", border=1, fill=True)
    pdf.cell(25, 8, "Count", border=1, fill=True)
    pdf.cell(45, 8, "Avg Confidence", border=1, fill=True)
    pdf.cell(50, 8, "Recommended Action", border=1, fill=True)
    pdf.ln()

    # Summary Rows with confidence stats
    pdf.set_font("Helvetica", "", 11)
    for dmg_type, count in damage_summary.items():
        action = "Immediate Repair" if "pothole" in dmg_type.lower() else "Monitor / Schedule"
        
        avg_conf = "N/A"
        if confidence_stats and dmg_type in confidence_stats:
            avg_conf = f"{np.mean(confidence_stats[dmg_type]):.2f}"
        
        pdf.cell(70, 8, dmg_type, border=1)
        pdf.cell(25, 8, str(count), border=1)
        pdf.cell(45, 8, avg_conf, border=1)
        pdf.cell(50, 8, action, border=1)
        pdf.ln()

    pdf.ln(10)

    # Footer Note
    pdf.set_font("Helvetica", "I", 10)
    pdf.cell(0, 8, "Note: This report is generated using automated road damage detection technology.", ln=True)
    pdf.cell(0, 8, "Confidence scores indicate model certainty (0.0-1.0, higher is better).", ln=True)

    # Export PDF
    pdf_output_path = os.path.join(tempfile.gettempdir(), f"roadbot_report_{uuid.uuid4().hex[:6]}.pdf")
    pdf.output(pdf_output_path)
    return pdf_output_path

# ---- CUSTOM CSS FOR MODERN STYLING ----
st.markdown("""
    <style>
        body {
            background-color: #f0f4f8;
            font-family: 'Segoe UI', sans-serif;
        }
        .main .block-container {
            padding-top: 2rem;
            background-color: #ffffff;
            border-radius: 10px;
            box-shadow: 0 4px 10px rgba(0,0,0,0.1);
        }
        h1 {
            color: #1c3d7c;
        }
        .stButton button {
            background-color: #1c3d7c;
            color: white;
            border-radius: 5px;
        }
        .detection-stats {
            background-color: #f8f9fa;
            padding: 15px;
            border-radius: 8px;
            border-left: 4px solid #1c3d7c;
            margin: 10px 0;
        }
        .video-container {
            border: 2px solid #e0e0e0;
            border-radius: 8px;
            padding: 10px;
            margin: 10px 0;
        }
        .image-container {
            border: 2px solid #e0e0e0;
            border-radius: 8px;
            padding: 10px;
            margin: 10px 0;
        }
    </style>
""", unsafe_allow_html=True)

# ---- MAIN APPLICATION ----
st.markdown("## 🚗 RoadBot: Detect and Classify Road Damage")
st.markdown("##### *Detect Fast. Repair Faster.*")

# ---- USER DETAILS FORM ----
st.subheader("Please enter your information")
with st.form(key='user_info_form'):
    name = st.text_input("Full Name")
    email = st.text_input("Email Address")
    location = st.text_input("Location of Inspection")
    submit_button = st.form_submit_button(label='Start Inspection')

if submit_button:
    if not name or not email or not location:
        st.warning("Please fill in all fields.")
    else:
        st.session_state['user_info'] = {
            "name": name,
            "email": email,
            "location": location
        }
        st.success(f"Hello {name}! You're inspecting road damage in {location}. Select input type below.")

# ---- INPUT TYPE SELECTION ----
if 'user_info' in st.session_state:
    st.subheader("Choose Input Type")
    input_type = st.radio("Select Input Type:", ("Image", "Video"))

    if input_type == "Image":
        uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])

        if uploaded_file is not None:
            original_image = Image.open(uploaded_file).convert("RGB")

            with st.spinner("🔍 Detecting road damage..."):
                detected_image, results = detect_image_damage(model, original_image, confidence_threshold=0.5)

                # Extract damage summary with confidence
                damage_counts = {}
                confidence_stats = {}
                
                if results and results[0].boxes is not None:
                    for box in results[0].boxes:
                        class_id = int(box.cls[0])
                        confidence = float(box.conf[0])
                        damage_type = model.names[class_id]
                        
                        damage_counts[damage_type] = damage_counts.get(damage_type, 0) + 1
                        
                        if damage_type not in confidence_stats:
                            confidence_stats[damage_type] = []
                        confidence_stats[damage_type].append(confidence)

            # Show both images side by side in containers
            st.markdown("### 📸 Image Analysis")
            st.markdown('<div class="image-container">', unsafe_allow_html=True)
            col1, col2 = st.columns(2)
            with col1:
                st.image(original_image, caption="Original Image", use_container_width=True)
            with col2:
                st.image(detected_image, caption="Detected Damage", use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

            # Enhanced damage summary with markdown styling
            st.markdown("### 📊 Detection Results")
            if damage_counts:
                # Create summary table
                summary_data = []
                for i, (damage_type, count) in enumerate(damage_counts.items(), 1):
                    avg_conf = np.mean(confidence_stats[damage_type])
                    summary_data.append({
                        "No.": i,
                        "Damage Type": damage_type,
                        "Total Detections": count,
                        "Average Confidence": f"{avg_conf:.3f}"
                    })
                
                # Display as a clean table
                df = pd.DataFrame(summary_data)
                st.dataframe(df, use_container_width=True)
                
                # Detailed stats in expandable section
                with st.expander("📈 View Detailed Statistics"):
                    for damage_type, confidences in confidence_stats.items():
                        st.markdown(f"**{damage_type}:**")
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("Total", len(confidences))
                        with col2:
                            st.metric("Avg Confidence", f"{np.mean(confidences):.3f}")
                        with col3:
                            st.metric("Max Confidence", f"{np.max(confidences):.3f}")
                        with col4:
                            st.metric("Min Confidence", f"{np.min(confidences):.3f}")
                        st.markdown("---")
                
                # Generate Report button
                if st.button("📋 Generate Report", use_container_width=True):
                    with st.spinner("Generating your PDF report..."):
                        pdf_path = generate_enhanced_pdf_report(
                            st.session_state['user_info'], 
                            detected_image, 
                            damage_counts,
                            confidence_stats
                        )
                        with open(pdf_path, "rb") as f:
                            st.download_button(
                                label="📥 Download PDF Report",
                                data=f,
                                file_name="roadbot_report.pdf",
                                mime="application/pdf",
                                use_container_width=True
                            )
            else:
                st.info("ℹ️ No damage detected with the current confidence threshold.")

    elif input_type == "Video":
        uploaded_file = st.file_uploader("Upload a video...", type=["mp4", "avi", "mov"])

        if uploaded_file is not None:
            # Show original video in container
            st.markdown("### 📹 Original Video")
            st.markdown('<div class="video-container">', unsafe_allow_html=True)
            st.video(uploaded_file)
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Process video with direct playback capability
            with st.spinner("🔍 Processing video and detecting damage..."):
                video_bytes, video_path, detection_stats = detect_video_damage_with_playback(
                    model, uploaded_file, confidence_threshold=0.5
                )

            # Check if processing was successful
            if video_bytes and video_path:
                st.success("✅ Detection completed!")
                
                # Display processed video with direct playback
                st.markdown("### 🎯 Processed Video with Damage Detection")
                st.markdown('<div class="video-container">', unsafe_allow_html=True)
                
                try:
                    # Direct video playback using the processed bytes
                    st.video(video_bytes)
                    st.success("✅ Video is ready to play!")
                    
                except Exception as e:
                    st.error(f"Error displaying video: {str(e)}")
                    st.info("📥 Video processing completed. Please use the download button below.")
                
                st.markdown('</div>', unsafe_allow_html=True)
                
                # Download button
                st.download_button(
                    label="📥 Download Processed Video",
                    data=video_bytes,
                    file_name="roadbot_processed_video.mp4",
                    mime="video/mp4",
                    use_container_width=True
                )
                
                # Show detection statistics
                if detection_stats:
                    st.markdown("### 📊 Detection Summary")
                    
                    # Create summary table
                    summary_data = []
                    for i, (damage_type, confidences) in enumerate(detection_stats.items(), 1):
                        summary_data.append({
                            "No.": i,
                            "Damage Type": damage_type,
                            "Total Detections": len(confidences),
                            "Average Confidence": f"{np.mean(confidences):.3f}"
                        })
                    
                    # Display as a clean table
                    df = pd.DataFrame(summary_data)
                    st.dataframe(df, use_container_width=True)
                    
                    # Detailed stats in expandable section
                    with st.expander("📈 View Detailed Statistics"):
                        for damage_type, confidences in detection_stats.items():
                            st.markdown(f"**{damage_type}:**")
                            col1, col2, col3, col4 = st.columns(4)
                            with col1:
                                st.metric("Total", len(confidences))
                            with col2:
                                st.metric("Avg Confidence", f"{np.mean(confidences):.3f}")
                            with col3:
                                st.metric("Max Confidence", f"{np.max(confidences):.3f}")
                            with col4:
                                st.metric("Min Confidence", f"{np.min(confidences):.3f}")
                            st.markdown("---")
                    
                    # Generate Report button
                    if st.button("📋 Generate Report", use_container_width=True):
                        with st.spinner("Generating your PDF report..."):
                            # Convert detection stats to damage counts for PDF
                            damage_counts = {k: len(v) for k, v in detection_stats.items()}
                            pdf_path = generate_enhanced_pdf_report(
                                st.session_state['user_info'], 
                                None,  # No image for video
                                damage_counts,
                                detection_stats
                            )
                            with open(pdf_path, "rb") as f:
                                st.download_button(
                                    label="📥 Download PDF Report",
                                    data=f,
                                    file_name="roadbot_report.pdf",
                                    mime="application/pdf",
                                    use_container_width=True
                                )
                else:
                    st.info("ℹ️ No damage detected in this video.")
                    
            else:
                st.error("❌ Video processing failed. Please try again with a different video file.")
                st.info("Make sure your video file is in a supported format (MP4, AVI, MOV) and not corrupted.")
