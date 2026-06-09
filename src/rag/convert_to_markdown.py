import os
import glob
import pdfplumber
import warnings
from pathlib import Path
from docx import Document
from docx.oxml.table import CT_Tbl
from docx.oxml.text.paragraph import CT_P
from docx.table import Table
from docx.text.paragraph import Paragraph

warnings.filterwarnings('ignore')

def iter_block_items(parent):
    """Duyệt qua các phần tử Paragraph và Table của tài liệu Word theo thứ tự xuất hiện."""
    parent_elm = parent.element.body if hasattr(parent, 'element') else parent._element
    for child in parent_elm.iterchildren():
        if isinstance(child, CT_P):
            yield Paragraph(child, parent)
        elif isinstance(child, CT_Tbl):
            yield Table(child, parent)

def docx_table_to_markdown(table):
    """Chuyển đổi một bảng trong python-docx thành bảng Markdown."""
    markdown_lines = []
    
    grid_data = []
    for row in table.rows:
        row_cells = []
        for cell in row.cells:
            text = cell.text.strip().replace('\n', ' ').replace('|', '\\|')
            row_cells.append(text)
        grid_data.append(row_cells)
        
    if not grid_data:
        return ""
        
    for idx, row in enumerate(grid_data):
        markdown_lines.append("| " + " | ".join(row) + " |")
        if idx == 0:
            markdown_lines.append("| " + " | ".join(["---"] * len(row)) + " |")
            
    return "\n".join(markdown_lines)

def convert_docx_to_markdown(docx_path, output_md_path):
    """Chuyển đổi toàn bộ file DOCX thành Markdown (giữ nguyên bảng)."""
    print(f" -> Đang chuyển đổi Word: {os.path.basename(docx_path)}...")
    doc = Document(docx_path)
    md_content = []
    
    for item in iter_block_items(doc):
        if isinstance(item, Paragraph):
            text = item.text.strip()
            if text:
                md_content.append(text + "\n")
        elif isinstance(item, Table):
            table_md = docx_table_to_markdown(item)
            if table_md:
                md_content.append("\n" + table_md + "\n")
                
    with open(output_md_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(md_content))
    print(f"      ✅ Đã lưu file Markdown tại: {output_md_path}")

# pdf utils
def is_char_in_bbox(char, bbox):
    """Kiểm tra xem một ký tự có nằm trong bounding box của bảng không."""
    x0, top, x1, bottom = bbox
    cx0 = char.get("x0", 0)
    cx1 = char.get("x1", 0)
    ctop = char.get("top", 0)
    cbottom = char.get("bottom", 0)
    
    return not (cx1 < x0 or cx0 > x1 or cbottom < top or ctop > bottom)

def pdf_table_to_markdown(table_data):
    """Chuyển đổi dữ liệu bảng thô từ pdfplumber thành bảng Markdown."""
    if not table_data or not table_data[0]:
        return ""
    
    markdown_lines = []
    # Làm sạch dữ liệu bảng
    cleaned_rows = []
    for row in table_data:
        cleaned_row = []
        for cell in row:
            val = str(cell or "").strip().replace('\n', ' ').replace('|', '\\|')
            cleaned_row.append(val)
        cleaned_rows.append(cleaned_row)
        
    for idx, row in enumerate(cleaned_rows):
        markdown_lines.append("| " + " | ".join(row) + " |")
        if idx == 0:
            markdown_lines.append("| " + " | ".join(["---"] * len(row)) + " |")
            
    return "\n".join(markdown_lines)

def convert_pdf_to_markdown(pdf_path, output_md_path):
    """Chuyển đổi file PDF thành Markdown bằng pdfplumber (lọc chữ trong bảng để chèn bảng Markdown sạch)."""
    print(f" -> Đang chuyển đổi PDF: {os.path.basename(pdf_path)}...")
    md_content = []
    
    with pdfplumber.open(pdf_path) as pdf:
        for page_idx, page in enumerate(pdf.pages):
            tables = page.find_tables()
            table_bboxes = [t.bbox for t in tables]
            
            # 1. Trích xuất plain text ngoài bảng
            # Lọc bỏ các ký tự nằm bên trong bounding box của bất kỳ bảng nào
            def char_filter(char):
                return not any(is_char_in_bbox(char, bbox) for bbox in table_bboxes)
                
            filtered_page = page.filter(char_filter)
            plain_text = filtered_page.extract_text() or ""
            
            md_content.append(f"<!-- Page {page_idx + 1} -->\n" + plain_text.strip() + "\n")
            
            # 2. Trích xuất bảng và chuyển thành Markdown
            for table in tables:
                table_data = table.extract()
                table_md = pdf_table_to_markdown(table_data)
                if table_md:
                    md_content.append("\n" + table_md + "\n")
                    
    with open(output_md_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(md_content))
    print(f"Đã lưu file Markdown tại: {output_md_path}")

# MAIN PIPELINE
def main():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(current_dir))
    data_dir = os.path.join(project_root, "data_RAG")
    
    if not os.path.exists(data_dir):
        print(f"Lỗi: Thư mục {data_dir} không tồn tại!")
        return

    pdf_files = glob.glob(os.path.join(data_dir, "*.pdf"))
    for pdf_file in pdf_files:
        out_path = pdf_file.replace(".pdf", ".md")
        convert_pdf_to_markdown(pdf_file, out_path)
        
    docx_files = glob.glob(os.path.join(data_dir, "*.docx"))
    for docx_file in docx_files:
        out_path = docx_file.replace(".docx", ".md")
        convert_docx_to_markdown(docx_file, out_path)

    print("Đã hoàn tất việc trích xuất văn bản và bảng biểu sang định dạng Markdown!")

if __name__ == "__main__":
    main()
