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

            # Display DataFrame statistics
            st.write("### File Statistics")
            # Total number of columns
            total_columns = len(df.columns)
            st.write(f"- **Total Number of Columns**: {total_columns}")
            # Total number of rows 
            total_rows = len(df['id_doc'])
            st.write(f"- **Total Number of Rows**: {total_rows}")
            # Check if 'label' column exists, and calculate label statistics
            if 'label' in df.columns:
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
            else:
                st.write("- **Label Information**: No 'label' column found in the file.")

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

if __name__ == "__main__":
    app()