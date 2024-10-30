import streamlit as st
import numpy as np

# دالة لتنسيق المصفوفة في نص بدون أقواس
def format_matrix(matrix):
    formatted_text = ""
    for row in matrix:
        formatted_text += " ".join(f"{val:8.2f}" for val in row) + "\n"
    return formatted_text

# دالة لحل النظام باستخدام شكل الصفوف المتجهة
def solve_matrix(matrix):
    try:
        matrix = np.array(matrix, dtype=float)  # تحويل المدخلات إلى مصفوفة numpy
        rows, columns = matrix.shape
        solution_text = "حل النظام باستخدام شكل الصفوف المتجهة:\n\n"
        leading_row = 0  # متغير لتتبع الصف الرائد

        # التحقق من التناقضات الأولية في النظام
        for i in range(rows):
            if np.all(matrix[i, :-1] == 0) and matrix[i, -1] != 0:
                return (
                    "لا توجد حلول لأن النظام يحتوي على تناقض (صف مع أصفار ولكن نتيجة غير صفرية).\n",
                    matrix
                )

        # التحقق مما إذا كان للنظام عدد كافٍ من المعادلات
        if rows < columns - 1:
            return "النظام غير صالح لأنه يحتوي على عدد أقل من المعادلات مقارنة بالمتغيرات.\n", matrix

        # خطوات تحويل المصفوفة إلى شكل الصفوف المتجهة
        for col in range(columns - 1):
            pivot_row = leading_row
            while pivot_row < rows and matrix[pivot_row, col] == 0:
                pivot_row += 1

            if pivot_row == rows:
                continue

            # تبديل الصفوف لجعل العنصر الرائد في الموضع الصحيح
            if pivot_row != leading_row:
                matrix[[leading_row, pivot_row]] = matrix[[pivot_row, leading_row]]
                solution_text += f"تبديل الصف {leading_row + 1} مع الصف {pivot_row + 1}\n"
                solution_text += format_matrix(matrix) + "\n"

            # جعل العنصر الرائد يساوي 1
            lead_element = matrix[leading_row, col]
            if lead_element != 0:
                matrix[leading_row] /= lead_element
                solution_text += f"قسمت جميع العناصر في الصف {leading_row + 1} على {lead_element:.2f} لجعل العنصر الرائد في العمود {col + 1} يساوي 1\n"
                solution_text += format_matrix(matrix) + "\n"

            # تصفير العناصر أسفل العنصر الرائد
            for i in range(leading_row + 1, rows):
                factor = matrix[i, col]
                matrix[i] -= factor * matrix[leading_row]
                if factor != 0:
                    solution_text += f"ضربت جميع العناصر في الصف {leading_row + 1} بـ {factor:.2f} وطرحتها من الصف {i + 1} لتصفير العنصر في العمود {col + 1}\n"
                    solution_text += format_matrix(matrix) + "\n"

            leading_row += 1

        # خطوات تحويل المصفوفة إلى شكل الصفوف المتجهة المنقحة
        for col in range(columns - 2, -1, -1):
            row = -1
            for i in range(rows):
                if matrix[i, col] == 1:
                    row = i
                    break

            if row != -1:
                for i in range(row):
                    factor = matrix[i, col]
                    matrix[i] -= factor * matrix[row]
                    if factor != 0:
                        solution_text += f"ضربت جميع العناصر في الصف {row + 1} بـ {factor:.2f} وطرحتها من الصف {i + 1} لتصفير العنصر في العمود {col + 1}\n"
                        solution_text += format_matrix(matrix) + "\n"

        # استخراج الحلول النهائية
        solutions = np.zeros(columns - 1)
        for i in range(rows - 1, -1, -1):
            if np.all(matrix[i, :-1] == 0):
                continue
            sum_val = matrix[i, -1]
            for j in range(i + 1, columns - 1):
                sum_val -= matrix[i, j] * solutions[j]
            solutions[i] = sum_val / matrix[i, i]

        solution_text += "\nالحلول النهائية:\n"
        for index, sol in enumerate(solutions):
            solution_text += f"x{index + 1} = {sol:.2f}\n"

        return solution_text
    except Exception as e:
        return f"حدث خطأ: {str(e)}"


# تعيين إعدادات الصفحة لتكون بالتخطيط الواسع
st.set_page_config(layout="wide")
st.title("حل نظام المعادلات الخطية باستخدام شكل الصفوف المتجهة")

# الحصول على عدد الصفوف والأعمدة من المستخدم
rows = st.number_input("أدخل عدد الصفوف (المعادلات):", min_value=1, max_value=10, value=3)
columns = st.number_input("أدخل عدد الأعمدة (المتغيرات + 1):", min_value=2, max_value=10, value=4)

# إدخال قيم المصفوفة من المستخدم باستخدام هيكل شبيه بالجدول
st.write("أدخل قيم المصفوفة:")
matrix = []
for i in range(int(rows)):
    row = []
    cols = st.columns(int(columns))  # إنشاء أعمدة بناءً على عدد الأعمدة المطلوبة
    for j in range(int(columns)):
        with cols[j]:  # وضع كل إدخال في الموضع المناسب في الجدول
            val = st.number_input(f"الصف {i+1}, العمود {j+1}", value=0.0, step=1.0, key=f"input_{i}_{j}")
            row.append(val)
    matrix.append(row)

# عرض المصفوفة المدخلة وتحليلها عند الضغط على زر "حل"
if st.button("حل", key="solve_button", help="انقر لحل المصفوفة"):
    # استخدام عرض كامل للزر
    button_style = """
    <style>
    .stButton > button {
        width: 100%;
        height: 50px;
        font-size: 20px;
    }
    </style>
    """
    st.markdown(button_style, unsafe_allow_html=True)
    
    solution_text = solve_matrix(matrix)
    st.subheader("النتيجة:")
    st.text(solution_text)
