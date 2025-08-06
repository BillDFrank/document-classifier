import streamlit as st
import pandas as pd
import io
import os
import json
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def app():
    """Handles the conversion of a Parquet file to CSV."""
    st.title("📊 Convert Parquet to CSV")
    st.write("Upload a Parquet file to convert it to CSV format.")

    # File uploader for Parquet files
    uploaded_file = st.file_uploader("Select a Parquet file", type=["parquet"])

    if uploaded_file:
        try:
            logger.info("Processing uploaded Parquet file")
            # Read the Parquet file
            df = pd.read_parquet(uploaded_file)
            st.success("✅ Parquet file loaded successfully!")
            st.write("Preview of the loaded file:")
            st.dataframe(df.head())

            # Display DataFrame statistics
            st.write("### 📈 File Statistics")
            try:
                # Total number of columns
                total_columns = len(df.columns)
                st.write(f"- **Total Number of Columns**: {total_columns}")
                
                # Total number of rows
                if 'id_doc' in df.columns:
                    total_rows = len(df['id_doc'])
                    st.write(f"- **Total Number of Rows**: {total_rows}")
                else:
                    total_rows = len(df)
                    st.write(f"- **Total Number of Rows**: {total_rows}")
                    logger.warning("'id_doc' column not found, using DataFrame length")
                
                # Check if 'label' column exists, and calculate label statistics
                if 'label' in df.columns:
                    try:
                        # Ensure labels are treated as strings and handle NaN/None
                        df['label'] = df['label'].fillna("").astype(str)
                        # Total number of rows with labels (non-empty strings)
                        labeled_rows = df[df['label'] != ""].shape[0]
                        st.write(f"- **Total Number of Rows with Labels**: {labeled_rows}")
                        # Count of each label (excluding empty strings)
                        label_counts = df[df['label'] != ""]['label'].value_counts()
                        if not label_counts.empty:
                            st.write("- **Label Distribution**:")
                            for label, count in label_counts.items():
                                st.write(f"  - {label}: {count}")
                        else:
                            st.write("- **Label Distribution**: No labels found.")
                    except Exception as e:
                        logger.error(f"Error processing label statistics: {e}")
                        st.error(f"Error processing label statistics: {e}")
                else:
                    st.write("- **Label Information**: No 'label' column found in the file.")
                    logger.info("No 'label' column found in the file")
                    
            except Exception as e:
                logger.error(f"Error calculating file statistics: {e}")
                st.error(f"Error calculating file statistics: {e}")

            # Convert DataFrame to CSV
            try:
                csv_buffer = io.StringIO()
                df.to_csv(csv_buffer, index=False)
                csv_data = csv_buffer.getvalue()
                logger.info("Successfully converted DataFrame to CSV format")

                # Provide a download button for the CSV file
                output_filename = st.text_input(
                    "Output CSV file name", value="converted_file.csv")
                if not output_filename.endswith(".csv"):
                    output_filename += ".csv"

                st.download_button(
                    label="📥 Download as CSV",
                    data=csv_data,
                    file_name=output_filename,
                    mime="text/csv"
                )
                logger.info(f"Download button created for file: {output_filename}")
                
            except Exception as e:
                logger.error(f"Error converting to CSV: {e}")
                st.error(f"Error converting to CSV: {e}")
                
        except Exception as e:
            logger.error(f"Error processing the Parquet file: {e}")
            st.error(f"Error processing the Parquet file: {e}")

if __name__ == "__main__":
    app()