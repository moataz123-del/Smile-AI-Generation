import streamlit as st
import helper as Helper
import face as Face
import os

st.set_page_config(layout="wide", page_title="Smile Design")
st.title("Smile Design")

# Initialize session state
if "file_path" not in st.session_state:
    st.session_state["file_path"] = None
if "analysis_done" not in st.session_state:
    st.session_state["analysis_done"] = False

# Sidebar for controls
with st.sidebar:
    st.header("Controls")
    
    # File upload
    file_path = Helper.browsefiles()
    if file_path:
        st.session_state["file_path"] = file_path
        st.success(f"File uploaded: {os.path.basename(file_path)}")
    
    # Analysis button
    if st.session_state["file_path"] and not st.session_state["analysis_done"]:
        if st.button("Analyze Smile (Check Principles)"):
            with st.spinner("Analyzing smile..."):
                try:
                    result = Helper.start(st.session_state["file_path"])
                    if result:
                        Face.checkAll()
                        st.session_state["analysis_done"] = True
                        st.success("Analysis complete!")
                except Exception as e:
                    st.error(f"Analysis failed: {str(e)}")
    
    # Reset button
    if st.button("Reset"):
        st.session_state["file_path"] = None
        st.session_state["analysis_done"] = False
        st.rerun()

# Main content area
col1, col2 = st.columns(2)

with col1:
    st.header("Original Image")
    if st.session_state["file_path"]:
        Helper.plotImage(st.session_state["file_path"], caption="Original Image")
    else:
        st.info("Please upload an image to begin")

with col2:
    st.header("Analysis Results")
    if st.session_state["analysis_done"]:
        # Show analysis results
        if hasattr(Face, 'results') and Face.results:
            st.text_area("Analysis Results:", Face.results, height=200)
        
        # Show processed image if available
        if os.path.exists(Face.finalImagePath):
            Helper.plotImage(Face.finalImagePath, caption="After Analysis")
        
        # Shape selection
        st.subheader("Face Shape Templates")
        shape = st.selectbox("Choose face shape", ["Square", "Rectangle", "Triangle", "Oval"], key="shapesComboBox")
        if shape:
            if st.button("Apply Template"):
                try:
                    Face.applyTemplateToResult(shape)
                    st.success("Template applied!")
                    Helper.plotImage(Face.finalImagePath, caption="After Template Application")
                except Exception as e:
                    st.error(f"Template application failed: {str(e)}")
        
        # Template scaling
        st.subheader("Template Scaling")
        col_scale1, col_scale2 = st.columns(2)
        with col_scale1:
            if st.button("Scale Template Up"):
                Face.scaleCurrentTemplate(1.1)
                st.info("Scale up applied")
                Helper.plotImage(Face.finalImagePath, caption="After Scaling")
        with col_scale2:
            if st.button("Scale Template Down"):
                Face.scaleCurrentTemplate(0.9)
                st.info("Scale down applied")
                Helper.plotImage(Face.finalImagePath, caption="After Scaling")
    else:
        st.info("Run analysis to see results")
