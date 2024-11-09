import streamlit as st
import numpy as np

# دالة لتحويل المصفوفة إلى نص منسق بدون أقواس
def format_matrix(matrix):
    formatted_text = ""
    for row in matrix:
        formatted_text += "[" + " ".join(f"{elem: .2f}" for elem in row)+ "]" + "\n"
    return formatted_text

def format_matrix_2(matrix):
    formatted_text = ""
    for row in matrix:
        mid_index = len(row) // 2  # Calculate the middle index
        formatted_row = []

        for i, elem in enumerate(row):
            formatted_row.append(f"{elem: .2f}")
            if i == mid_index - 1:  # Add '|' after the middle item if it exists
                formatted_row.append("|")

        formatted_text += "[" + " ".join(formatted_row) + "]" + "\n"
    return formatted_text


# دالة حل النظام باستخدام تحويل الصفوف إلى Row Echelon Form
def solve_matrix(matrix):
    try:
        matrix = np.array(matrix, dtype=float)
        rows, columns = matrix.shape
        solution_text = "حل النظام باستخدام تحويل الصفوف (Row Echelon Form):\n\n"
        leading_row = 0

        for i in range(rows):
            if np.all(matrix[i, :-1] == 0) and matrix[i, -1] != 0:
                return "لا يوجد حل لأن النظام يحتوي على تناقض.\n", matrix

        if rows < columns - 1:
            return "النظام غير صالح لأنه يحتوي على عدد معادلات أقل من عدد المتغيرات.\n", matrix

        for col in range(columns - 1):
            pivot_row = leading_row
            while pivot_row < rows and matrix[pivot_row, col] == 0:
                pivot_row += 1

            if pivot_row == rows:
                continue

            if pivot_row != leading_row:
                matrix[[leading_row, pivot_row]] = matrix[[pivot_row, leading_row]]
                solution_text += f"تم تبديل الصف {leading_row + 1} مع الصف {pivot_row + 1}\n"
                solution_text += format_matrix(matrix) + "\n"

            lead_element = matrix[leading_row, col]
            if lead_element != 0:
                matrix[leading_row] /= lead_element
                solution_text += f"تم قسمة جميع عناصر الصف {leading_row + 1} على {lead_element:.2f}\n"
                solution_text += format_matrix(matrix) + "\n"

            for i in range(leading_row + 1, rows):
                factor = matrix[i, col]
                matrix[i] -= factor * matrix[leading_row]
                if factor != 0:
                    solution_text += f"تم طرح {factor:.2f} * الصف {leading_row + 1} من الصف {i + 1}\n"
                    solution_text += format_matrix(matrix) + "\n"

            leading_row += 1

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

# دالة لجمع مصفوفتين
def add_matrices(matrix1, matrix2):
    try:
        result = np.add(matrix1, matrix2)
        return result
    except Exception as e:
        return f"حدث خطأ: {str(e)}"

# دالة لطرح مصفوفتين
def subtract_matrices(matrix1, matrix2):
    try:
        result = np.subtract(matrix1, matrix2)
        return result
    except Exception as e:
        return f"حدث خطأ: {str(e)}"

# دالة لضرب مصفوفتين
def multiply_matrices(matrix1, matrix2):
    try:
        result = np.dot(matrix1, matrix2)
        return result
    except ValueError:
        return "لا يمكن ضرب المصفوفات لأن عدد أعمدة المصفوفة الأولى لا يساوي عدد صفوف المصفوفة الثانية."
    except Exception as e:
        return f"حدث خطأ: {str(e)}"

def transpose_matrix(matrix):
    try:
        result = np.transpose(matrix)
        return result
    except Exception as e:
        return f"حدث خطأ: {str(e)}"

def invert_matrix(matrix):
    try:
        matrix = np.array(matrix, dtype=float)
        rows, columns = matrix.shape
        
        if rows != columns:
            return "لا يمكن عكس المصفوفة لأنها غير مربعة."

        # إنشاء المصفوفة المعززة
        augmented_matrix = np.hstack((matrix, np.eye(rows)))
        solution_text = "عكس المصفوفة باستخدام Gauss-Jordan Elimination:\n\n"

        # تطبيق عمليات الصف
        leading_row = 0
        for col in range(columns):
            pivot_row = leading_row
            while pivot_row < rows and augmented_matrix[pivot_row, col] == 0:
                pivot_row += 1

            if pivot_row == rows:
                continue

            if pivot_row != leading_row:
                augmented_matrix[[leading_row, pivot_row]] = augmented_matrix[[pivot_row, leading_row]]
                solution_text += f"تم تبديل الصف {leading_row + 1} مع الصف {pivot_row + 1}\n"
                solution_text += format_matrix_2(augmented_matrix) + "\n"

            lead_element = augmented_matrix[leading_row, col]
            if lead_element != 0:
                augmented_matrix[leading_row] /= lead_element
                solution_text += f"تم قسمة جميع عناصر الصف {leading_row + 1} على {lead_element:.2f}\n"
                solution_text += format_matrix_2(augmented_matrix) + "\n"

            for i in range(rows):
                if i != leading_row:
                    factor = augmented_matrix[i, col]
                    augmented_matrix[i] -= factor * augmented_matrix[leading_row]
                    if factor != 0:
                        solution_text += f"تم طرح {factor:.2f} * الصف {leading_row + 1} من الصف {i + 1}\n"
                        solution_text += format_matrix_2(augmented_matrix) + "\n"

            leading_row += 1
            if leading_row == rows:
                break

        # استخراج المصفوفة المعكوسة
        inverse_matrix = augmented_matrix[:, columns:]
        return solution_text, inverse_matrix
    except Exception as e:
        return f"حدث خطأ: {str(e)}", None

# واجهة المستخدم باستخدام Streamlit
st.sidebar.title("اختيار الصفحة")
page = st.sidebar.selectbox("اختر الصفحة:", ["حل نظام المعادلات", "جمع المصفوفات", "طرح المصفوفات", "ضرب المصفوفات", "ترانسبوز المصفوفة","عكس المصفوفة"])

button_style = """
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Cairo:wght@200..1000&family=Sofia+Sans:ital,wght@0,1..1000;1,1..1000&display=swap" rel="stylesheet">
    <style>
    *{
        font-family: 'Cairo', sans-serif !important;
    }
    .stButton > button {
        width: 100%;
        height: 50px;
        font-size: 20px;
    }
    </style>
    """

st.markdown(button_style, unsafe_allow_html=True)


if page == "حل نظام المعادلات":
    st.title("حل نظام المعادلات الخطية باستخدام Row Echelon Form")
    rows = st.number_input("أدخل عدد الصفوف:", min_value=1, max_value=10, value=3)
    columns = st.number_input("أدخل عدد الأعمدة:", min_value=2, max_value=10, value=4)

    st.write("أدخل قيم المصفوفة:")
    matrix = []
    for i in range(int(rows)):
        row = []
        cols = st.columns(int(columns))
        for j in range(int(columns)):
            with cols[j]:
                val = st.number_input(f"الصف {i + 1}, العمود {j + 1}", value=0.0, step=1.0, key=f"input_{i}_{j}")
                row.append(val)
        matrix.append(row)

    if st.button("حل"):
        solution_text = solve_matrix(matrix)
        st.subheader("الناتج:")
        st.text(solution_text)

elif page == "جمع المصفوفات":
    st.title("جمع المصفوفات")
    rows = st.number_input("أدخل عدد الصفوف:", min_value=1, max_value=10, value=3)
    columns = st.number_input("أدخل عدد الأعمدة:", min_value=1, max_value=10, value=3)

    st.write("أدخل قيم المصفوفة الأولى:")
    matrix1 = []
    for i in range(int(rows)):
        row = []
        cols = st.columns(int(columns))
        for j in range(int(columns)):
            with cols[j]:
                val = st.number_input(f"المصفوفة الأولى - الصف {i + 1}, العمود {j + 1}", value=0.0, step=1.0, key=f"matrix1_{i}_{j}")
                row.append(val)
        matrix1.append(row)

    st.write("أدخل قيم المصفوفة الثانية:")
    matrix2 = []
    for i in range(int(rows)):
        row = []
        cols = st.columns(int(columns))
        for j in range(int(columns)):
            with cols[j]:
                val = st.number_input(f"المصفوفة الثانية - الصف {i + 1}, العمود {j + 1}", value=0.0, step=1.0, key=f"matrix2_{i}_{j}")
                row.append(val)
        matrix2.append(row)

    if st.button("جمع المصفوفات"):
        result = add_matrices(matrix1, matrix2)
        st.subheader("الناتج:")
        st.text(format_matrix(result))

elif page == "طرح المصفوفات":
    st.title("طرح المصفوفات")
    rows = st.number_input("أدخل عدد الصفوف:", min_value=1, max_value=10, value=3)
    columns = st.number_input("أدخل عدد الأعمدة:", min_value=1, max_value=10, value=3)

    st.write("أدخل قيم المصفوفة الأولى:")
    matrix1 = []
    for i in range(int(rows)):
        row = []
        cols = st.columns(int(columns))
        for j in range(int(columns)):
            with cols[j]:
                val = st.number_input(f"المصفوفة الأولى - الصف {i + 1}, العمود {j + 1}", value=0.0, step=1.0, key=f"matrix1_{i}_{j}")
                row.append(val)
        matrix1.append(row)

    st.write("أدخل قيم المصفوفة الثانية:")
    matrix2 = []
    for i in range(int(rows)):
        row = []
        cols = st.columns(int(columns))
        for j in range(int(columns)):
            with cols[j]:
                val = st.number_input(f"المصفوفة الثانية - الصف {i + 1}, العمود {j + 1}", value=0.0, step=1.0, key=f"matrix2_{i}_{j}")
                row.append(val)
        matrix2.append(row)

    if st.button("طرح المصفوفات"):
        result = subtract_matrices(matrix1, matrix2)
        st.subheader("الناتج:")
        st.text(format_matrix(result))

elif page == "ضرب المصفوفات":
    st.title("ضرب المصفوفات")
    rows1 = st.number_input("أدخل عدد الصفوف للمصفوفة الأولى:", min_value=1, max_value=10, value=3)
    columns1 = st.number_input("أدخل عدد الأعمدة للمصفوفة الأولى:", min_value=1, max_value=10, value=3)

    rows2 = columns1
    columns2 = st.number_input("أدخل عدد الأعمدة للمصفوفة الثانية:", min_value=1, max_value=10, value=3)

    st.write("أدخل قيم المصفوفة الأولى:")
    matrix1 = []
    for i in range(int(rows1)):
        row = []
        cols = st.columns(int(columns1))
        for j in range(int(columns1)):
            with cols[j]:
                val = st.number_input(f"المصفوفة الأولى - الصف {i + 1}, العمود {j + 1}", value=0.0, step=1.0, key=f"matrix1_{i}_{j}")
                row.append(val)
        matrix1.append(row)

    # إدخال قيم المصفوفة الثانية من المستخدم
    st.write("أدخل قيم المصفوفة الثانية:")
    matrix2 = []
    for i in range(int(rows2)):
        row = []
        cols = st.columns(int(columns2))
        for j in range(int(columns2)):
            with cols[j]:
                val = st.number_input(f"المصفوفة الثانية - الصف {i + 1}, العمود {j + 1}", value=0.0, step=1.0, key=f"matrix2_{i}_{j}")
                row.append(val)
        matrix2.append(row)

    # تحويل القوائم إلى مصفوفات numpy
    matrix1 = np.array(matrix1)
    matrix2 = np.array(matrix2)

    # ضرب المصفوفات وعرض النتيجة عند الضغط على زر "ضرب"
    if st.button("ضرب المصفوفات"):
        result = multiply_matrices(matrix1, matrix2)
        st.subheader("الناتج:")
        if isinstance(result, str):  # في حال كانت النتيجة رسالة خطأ
            st.text(result)
        else:
            st.text(format_matrix(result))
            
elif page == "ترانسبوز المصفوفة":
    st.title("ترانسبوز المصفوفة")
    rows = st.number_input("أدخل عدد الصفوف:", min_value=1, max_value=10, value=3)
    columns = st.number_input("أدخل عدد الأعمدة:", min_value=1, max_value=10, value=3)

    st.write("أدخل قيم المصفوفة:")
    matrix = []
    for i in range(int(rows)):
        row = []
        cols = st.columns(int(columns))
        for j in range(int(columns)):
            with cols[j]:
                val = st.number_input(f"الصف {i + 1}, العمود {j + 1}", value=0.0, step=1.0, key=f"input_{i}_{j}")
                row.append(val)
        matrix.append(row)

    if st.button("ترانسبوز المصفوفة"):
        result = transpose_matrix(matrix)
        st.subheader("الناتج:")
        st.text(format_matrix(result))
        
elif page == "عكس المصفوفة":
    st.title("عكس المصفوفة باستخدام Gauss-Jordan Elimination")
    rows = st.number_input("أدخل عدد الصفوف:", min_value=1, max_value=10, value=3)
    columns = st.number_input("أدخل عدد الأعمدة:", min_value=1, max_value=10, value=3)

    st.write("أدخل قيم المصفوفة:")
    matrix = []
    for i in range(int(rows)):
        row = []
        cols = st.columns(int(columns))
        for j in range(int(columns)):
            with cols[j]:
                val = st.number_input(f"الصف {i + 1}, العمود {j + 1}", value=0.0, step=1.0, key=f"input_{i}_{j}")
                row.append(val)
        matrix.append(row)

    if st.button("عكس المصفوفة"):
        solution_text, inverse_matrix = invert_matrix(matrix)
        st.subheader("الناتج:")
        if inverse_matrix is not None:
            st.text(solution_text)
            st.text("المصفوفة المعكوسة:")
            st.text(format_matrix(inverse_matrix))
        else:
            st.text(solution_text)
