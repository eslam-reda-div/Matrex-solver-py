import streamlit as st
import numpy as np

# دالة حل النظام باستخدام تحويل الصفوف إلى Row Echelon Form
def solve_matrix(matrix):
    try:
        matrix = np.array(matrix, dtype=float)  # تحويل المدخلات إلى مصفوفة numpy
        rows, columns = matrix.shape
        solution_text = "حل النظام باستخدام تحويل الصفوف (Row Echelon Form):\n\n"
        leading_row = 0  # متغير لتتبع الصف الرئيسي
        step_matrices = []  # لتخزين حالة المصفوفة بعد كل خطوة

        # التحقق من وجود تناقضات أولية في النظام
        for i in range(rows):
            if np.all(matrix[i, :-1] == 0) and matrix[i, -1] != 0:
                return (
                    "لا يوجد حل لأن النظام يحتوي على تناقض (صف يحتوي على أصفار ولكن النتيجة غير صفرية).\n",
                    matrix
                )

        # التحقق من أن النظام يحتوي على معادلات كافية
        if rows < columns - 1:
            return "النظام غير صالح لأنه يحتوي على عدد معادلات أقل من عدد المتغيرات.\n", matrix

        # الخطوات لجعل المصفوفة في صورة Row Echelon Form
        for col in range(columns - 1):
            pivot_row = leading_row
            while pivot_row < rows and matrix[pivot_row, col] == 0:
                pivot_row += 1

            if pivot_row == rows:
                continue

            # تبديل الصفوف لجعل الصف ذو العنصر الرئيسي في الموقع الصحيح
            if pivot_row != leading_row:
                matrix[[leading_row, pivot_row]] = matrix[[pivot_row, leading_row]]
                solution_text += f"تم تبديل الصف {leading_row + 1} مع الصف {pivot_row + 1}\n\n"
                step_matrices.append(matrix.copy())  # تخزين حالة المصفوفة

            # جعل العنصر الرئيسي يساوي 1
            lead_element = matrix[leading_row, col]
            if lead_element != 0:
                matrix[leading_row] /= lead_element
                solution_text += f"تم قسمة جميع عناصر الصف {leading_row + 1} على {lead_element:.2f} لجعل العنصر الرئيسي في العمود {col + 1} يساوي 1\n\n"
                step_matrices.append(matrix.copy())  # تخزين حالة المصفوفة

            # تصفير العناصر أسفل العنصر الرئيسي
            for i in range(leading_row + 1, rows):
                factor = matrix[i, col]
                matrix[i] -= factor * matrix[leading_row]
                if factor != 0:
                    solution_text += f"تم ضرب جميع عناصر الصف {leading_row + 1} في {factor:.2f} وطرحها من الصف {i + 1} لتصفير العنصر في العمود {col + 1}\n\n"
                    step_matrices.append(matrix.copy())  # تخزين حالة المصفوفة

            leading_row += 1

        # الخطوات لجعل المصفوفة في صورة Reduced Row Echelon Form
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
                        solution_text += f"تم ضرب جميع عناصر الصف {row + 1} في {factor:.2f} وطرحها من الصف {i + 1} لتصفير العنصر في العمود {col + 1}\n\n"
                        step_matrices.append(matrix.copy())  # تخزين حالة المصفوفة

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

        return solution_text, step_matrices
    except Exception as e:
        return f"حدث خطأ: {str(e)}", None


# واجهة المستخدم باستخدام Streamlit
st.title("حل نظام المعادلات الخطية باستخدام Row Echelon Form")

# الحصول على عدد الصفوف والأعمدة من المستخدم
rows = st.number_input("أدخل عدد الصفوف (المعادلات):", min_value=1, max_value=10, value=3)
columns = st.number_input("أدخل عدد الأعمدة (المتغيرات + 1):", min_value=2, max_value=10, value=4)

# إدخال قيم المصفوفة من المستخدم باستخدام شكل يشبه الجدول
st.write("أدخل قيم المصفوفة:")
matrix = []
for i in range(int(rows)):
    row = []
    cols = st.columns(int(columns))  # إنشاء أعمدة حسب عدد الأعمدة المطلوبة
    for j in range(int(columns)):
        with cols[j]:  # وضع كل عنصر إدخال في مكانه المناسب في الجدول
            val = st.number_input(f"الصف {i+1}, العمود {j+1}", value=0.0, step=1.0, key=f"input_{i}_{j}")
            row.append(val)
    matrix.append(row)

# عرض المصفوفة المدخلة وتحليلها عند الضغط على زر "حل"
if st.button("حل"):
    solution_text, step_matrices = solve_matrix(matrix)
    st.subheader("الناتج:")
    st.text(solution_text)
    
    if step_matrices:
        st.subheader("المصفوفة بعد كل خطوة:")
        for i, step_matrix in enumerate(step_matrices):
            st.write(f"الخطوة {i + 1}:")
            st.write(step_matrix)