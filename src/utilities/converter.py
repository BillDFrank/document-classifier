import streamlit as st
import pandas as pd
import io


def app():
    """Handles the conversion of a Parquet file to CSV."""
    st.title("Convert Parquet to CSV")
    st.write("Upload a Parquet file to convert it to CSV format.")

    # File uploader for Parquet files
    uploaded_file = st.file_uploader("Select a Parquet file", type=["parquet"])

    if uploaded_file:
        try:
            # Read the Parquet file
            df = pd.read_parquet(uploaded_file)
            st.success("Parquet file loaded successfully!")
            st.write("Preview of the loaded file:")
            st.dataframe(df.head())

            # Convert DataFrame to CSV
            csv_buffer = io.StringIO()
            df.to_csv(csv_buffer, index=False)
            csv_data = csv_buffer.getvalue()

            # Provide a download button for the CSV file
            output_filename = st.text_input(
                "Output CSV file name", value="converted_file.csv")
            if not output_filename.endswith(".csv"):
                output_filename += ".csv"

            st.download_button(
                label="Download as CSV",
                data=csv_data,
                file_name=output_filename,
                mime="text/csv"
            )
        except Exception as e:
            st.error(f"Error processing the Parquet file: {e}")
