import streamlit as st
import numpy as np

# دالة بتحول المصفوفة لنص منسق من غير أقواس زيادة 
def format_matrix(matrix):
    try:
        formatted_text = ""
        for row in matrix:
            formatted_text += "[" + " ".join(f"{elem: .2f}" for elem in row)+ "]" + "\n"
        return formatted_text
    except Exception as e:
        return f"حدث خطأ في تنسيق المصفوفة: {str(e)}"

# دالة بتحول المصفوفة لنص منسق وبتحط خط عمودي في النص
def format_matrix_2(matrix):
    try:
        formatted_text = ""
        for row in matrix:
            mid_index = len(row) // 2  # بنحسب النص بتاع الصف
            formatted_row = []

            for i, elem in enumerate(row):
                formatted_row.append(f"{elem: .2f}")
                if i == mid_index - 1:  # بنحط الخط العمودي | بعد العنصر اللي في النص
                    formatted_row.append("|")

            formatted_text += "[" + " ".join(formatted_row) + "]" + "\n"
        return formatted_text
    except Exception as e:
        return f"حدث خطأ في تنسيق المصفوفة: {str(e)}"

# دالة بتحل النظام باستخدام طريقة الصفوف المتدرجة
def solve_matrix(matrix):
    try:
        # التحقق من صحة المدخلات
        if not matrix or not all(len(row) == len(matrix[0]) for row in matrix):
            return "خطأ: المصفوفة غير صالحة. تأكد من أن جميع الصفوف لها نفس الطول."

        matrix = np.array(matrix, dtype=float)
        rows, columns = matrix.shape

        # التحقق من أن المصفوفة ليست فارغة
        if rows == 0 or columns == 0:
            return "خطأ: المصفوفة فارغة."

        solution_text = "حل النظام باستخدام تحويل الصفوف (Row Echelon Form):\n\n"
        leading_row = 0

        # بنشوف لو في صف كله اصفار والعنصر الاخير مش صفر يبقى مفيش حل
        for i in range(rows):
            if np.all(matrix[i, :-1] == 0) and matrix[i, -1] != 0:
                return "لا يوجد حل لأن النظام يحتوي على تناقض.\n", matrix

        # بنتأكد ان عدد المعادلات مش اقل من عدد المتغيرات
        if rows < columns - 1:
            return "النظام غير صالح لأنه يحتوي على عدد معادلات أقل من عدد المتغيرات.\n", matrix

        # التحقق من وجود قيم غير محددة
        if np.any(np.isnan(matrix)) or np.any(np.isinf(matrix)):
            return "خطأ: المصفوفة تحتوي على قيم غير صالحة (NaN أو Inf)."

        # بنمشي على كل عمود ونعمل العمليات المطلوبة
        for col in range(columns - 1):
            pivot_row = leading_row
            while pivot_row < rows and matrix[pivot_row, col] == 0:
                pivot_row += 1

            if pivot_row == rows:
                continue

            # بنبدل الصفوف لو لازم
            if pivot_row != leading_row:
                matrix[[leading_row, pivot_row]] = matrix[[pivot_row, leading_row]]
                solution_text += f"تم تبديل الصف {leading_row + 1} مع الصف {pivot_row + 1}\n"
                solution_text += format_matrix(matrix) + "\n"

            # بنقسم الصف على العنصر الرئيسي
            lead_element = matrix[leading_row, col]
            if lead_element != 0:
                if abs(lead_element) < 1e-10:  # التحقق من القسمة على أرقام صغيرة جداً
                    return "خطأ: محاولة القسمة على رقم قريب جداً من الصفر."
                matrix[leading_row] /= lead_element
                solution_text += f"تم قسمة جميع عناصر الصف {leading_row + 1} على {lead_element:.2f}\n"
                solution_text += format_matrix(matrix) + "\n"

            # بنطرح من باقي الصفوف
            for i in range(leading_row + 1, rows):
                factor = matrix[i, col]
                matrix[i] -= factor * matrix[leading_row]
                if factor != 0:
                    solution_text += f"تم طرح {factor:.2f} * الصف {leading_row + 1} من الصف {i + 1}\n"
                    solution_text += format_matrix(matrix) + "\n"

            leading_row += 1

        # بنحل المعادلات من تحت لفوق
        solutions = np.zeros(columns - 1)
        for i in range(rows - 1, -1, -1):
            if np.all(matrix[i, :-1] == 0):
                continue
            sum_val = matrix[i, -1]
            for j in range(i + 1, columns - 1):
                sum_val -= matrix[i, j] * solutions[j]
            if abs(matrix[i, i]) < 1e-10:  # التحقق من القسمة على أرقام صغيرة جداً
                return "خطأ: محاولة القسمة على رقم قريب جداً من الصفر."
            solutions[i] = sum_val / matrix[i, i]

        # التحقق من صحة الحلول
        if np.any(np.isnan(solutions)) or np.any(np.isinf(solutions)):
            return "خطأ: الحل يحتوي على قيم غير صالحة (NaN أو Inf)."

        # بنكتب الحلول النهائية
        solution_text += "\nالحلول النهائية:\n"
        for index, sol in enumerate(solutions):
            solution_text += f"x{index + 1} = {sol:.2f}\n"

        return solution_text
    except np.linalg.LinAlgError:
        return "خطأ: مشكلة في العمليات الجبرية. المصفوفة قد تكون غير قابلة للحل."
    except ValueError as e:
        return f"خطأ في القيم المدخلة: {str(e)}"
    except Exception as e:
        return f"حدث خطأ غير متوقع: {str(e)}"

# دالة بتجمع مصفوفتين مع بعض
def add_matrices(matrix1, matrix2):
    try:
        # التحقق من صحة المدخلات
        if not matrix1 or not matrix2:
            return "خطأ: المصفوفات فارغة."
        
        if len(matrix1) != len(matrix2) or len(matrix1[0]) != len(matrix2[0]):
            return "خطأ: المصفوفات يجب أن تكون من نفس الحجم."

        matrix1 = np.array(matrix1, dtype=float)
        matrix2 = np.array(matrix2, dtype=float)

        # التحقق من وجود قيم غير محددة
        if np.any(np.isnan(matrix1)) or np.any(np.isinf(matrix1)) or \
           np.any(np.isnan(matrix2)) or np.any(np.isinf(matrix2)):
            return "خطأ: المصفوفات تحتوي على قيم غير صالحة (NaN أو Inf)."

        result = np.add(matrix1, matrix2)
        return result
    except ValueError as e:
        return f"خطأ في القيم المدخلة: {str(e)}"
    except Exception as e:
        return f"حدث خطأ: {str(e)}"

# دالة بتطرح مصفوفتين من بعض
def subtract_matrices(matrix1, matrix2):
    try:
        # التحقق من صحة المدخلات
        if not matrix1 or not matrix2:
            return "خطأ: المصفوفات فارغة."
        
        if len(matrix1) != len(matrix2) or len(matrix1[0]) != len(matrix2[0]):
            return "خطأ: المصفوفات يجب أن تكون من نفس الحجم."

        matrix1 = np.array(matrix1, dtype=float)
        matrix2 = np.array(matrix2, dtype=float)

        # التحقق من وجود قيم غير محددة
        if np.any(np.isnan(matrix1)) or np.any(np.isinf(matrix1)) or \
           np.any(np.isnan(matrix2)) or np.any(np.isinf(matrix2)):
            return "خطأ: المصفوفات تحتوي على قيم غير صالحة (NaN أو Inf)."

        result = np.subtract(matrix1, matrix2)
        return result
    except ValueError as e:
        return f"خطأ في القيم المدخلة: {str(e)}"
    except Exception as e:
        return f"حدث خطأ: {str(e)}"

# دالة بتضرب مصفوفتين في بعض
def multiply_matrices(matrix1, matrix2):
    try:
        # التحقق من صحة المدخلات
        if not matrix1 or not matrix2:
            return "خطأ: المصفوفات فارغة."

        matrix1 = np.array(matrix1, dtype=float)
        matrix2 = np.array(matrix2, dtype=float)

        # التحقق من إمكانية الضرب
        if matrix1.shape[1] != matrix2.shape[0]:
            return "خطأ: لا يمكن ضرب المصفوفات. عدد أعمدة المصفوفة الأولى يجب أن يساوي عدد صفوف المصفوفة الثانية."

        # التحقق من وجود قيم غير محددة
        if np.any(np.isnan(matrix1)) or np.any(np.isinf(matrix1)) or \
           np.any(np.isnan(matrix2)) or np.any(np.isinf(matrix2)):
            return "خطأ: المصفوفات تحتوي على قيم غير صالحة (NaN أو Inf)."

        result = np.dot(matrix1, matrix2)
        return result
    except ValueError as e:
        return f"خطأ في القيم المدخلة: {str(e)}"
    except Exception as e:
        return f"حدث خطأ: {str(e)}"

# دالة بتعمل ترانسبوز للمصفوفة (بتقلب الصفوف والاعمدة)
def transpose_matrix(matrix):
    try:
        # التحقق من صحة المدخلات
        if not matrix:
            return "خطأ: المصفوفة فارغة."

        matrix = np.array(matrix, dtype=float)

        # التحقق من وجود قيم غير محددة
        if np.any(np.isnan(matrix)) or np.any(np.isinf(matrix)):
            return "خطأ: المصفوفة تحتوي على قيم غير صالحة (NaN أو Inf)."

        result = np.transpose(matrix)
        return result
    except ValueError as e:
        return f"خطأ في القيم المدخلة: {str(e)}"
    except Exception as e:
        return f"حدث خطأ: {str(e)}"

# دالة بتعكس المصفوفة باستخدام طريقة جاوس جوردان
def invert_matrix(matrix):
    try:
        # التحقق من صحة المدخلات
        if not matrix:
            return "خطأ: المصفوفة فارغة.", None

        matrix = np.array(matrix, dtype=float)
        rows, columns = matrix.shape
        
        # بنتأكد ان المصفوفة مربعة
        if rows != columns:
            return "خطأ: لا يمكن عكس المصفوفة لأنها غير مربعة.", None

        # التحقق من أن المحدد لا يساوي صفر
        if abs(np.linalg.det(matrix)) < 1e-10:
            return "خطأ: لا يمكن عكس المصفوفة لأن محددها يساوي صفر.", None

        # التحقق من وجود قيم غير محددة
        if np.any(np.isnan(matrix)) or np.any(np.isinf(matrix)):
            return "خطأ: المصفوفة تحتوي على قيم غير صالحة (NaN أو Inf).", None

        # بنعمل المصفوفة المعززة (المصفوفة الاصلية + مصفوفة الوحدة)
        augmented_matrix = np.hstack((matrix, np.eye(rows)))
        solution_text = "عكس المصفوفة باستخدام Gauss-Jordan Elimination:\n\n"

        # بنطبق عمليات الصفوف
        leading_row = 0
        for col in range(columns):
            pivot_row = leading_row
            while pivot_row < rows and augmented_matrix[pivot_row, col] == 0:
                pivot_row += 1

            if pivot_row == rows:
                continue

            # بنبدل الصفوف لو لازم
            if pivot_row != leading_row:
                augmented_matrix[[leading_row, pivot_row]] = augmented_matrix[[pivot_row, leading_row]]
                solution_text += f"تم تبديل الصف {leading_row + 1} مع الصف {pivot_row + 1}\n"
                solution_text += format_matrix_2(augmented_matrix) + "\n"

            # بنقسم الصف على العنصر الرئيسي
            lead_element = augmented_matrix[leading_row, col]
            if abs(lead_element) < 1e-10:  # التحقق من القسمة على أرقام صغيرة جداً
                return "خطأ: محاولة القسمة على رقم قريب جداً من الصفر.", None

            if lead_element != 0:
                augmented_matrix[leading_row] /= lead_element
                solution_text += f"تم قسمة جميع عناصر الصف {leading_row + 1} على {lead_element:.2f}\n"
                solution_text += format_matrix_2(augmented_matrix) + "\n"

            # بنطرح من باقي الصفوف
            for i in range(rows):
                if i != leading_row:
                    factor = augmented_matrix[i, col]
                    augmented_matrix[i] -= factor * augmented_matrix[leading_row]
                    if factor != 0:
                        solution_text += f"تم طرح {factor:.2f} مضروب في الصف {leading_row + 1} من الصف {i + 1}\n"
                        solution_text += format_matrix_2(augmented_matrix) + "\n"

            leading_row += 1
            if leading_row == rows:
                break

        # بناخد المصفوفة المعكوسة من النص التاني
        inverse_matrix = augmented_matrix[:, columns:]

        # التحقق من صحة المصفوفة المعكوسة
        if np.any(np.isnan(inverse_matrix)) or np.any(np.isinf(inverse_matrix)):
            return "خطأ: المصفوفة المعكوسة تحتوي على قيم غير صالحة (NaN أو Inf).", None

        return solution_text, inverse_matrix
    except np.linalg.LinAlgError:
        return "خطأ: مشكلة في العمليات الجبرية. المصفوفة قد تكون غير قابلة للعكس.", None
    except ValueError as e:
        return f"خطأ في القيم المدخلة: {str(e)}", None
    except Exception as e:
        return f"حدث خطأ غير متوقع: {str(e)}", None

# بنعمل واجهة المستخدم باستخدام Streamlit
st.sidebar.title("اختيار الصفحة")
page = st.sidebar.selectbox("اختر الصفحة:", ["حل نظام المعادلات", "جمع المصفوفات", "طرح المصفوفات", "ضرب المصفوفات", "ترانسبوز المصفوفة","عكس المصفوفة"])

# بنضيف ستايل للازرار والخطوط
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

try:
    # صفحة حل نظام المعادلات
    if page == "حل نظام المعادلات":
        st.title("حل نظام المعادلات الخطية باستخدام Row Echelon Form")
        
        # التحقق من صحة المدخلات
        try:
            rows = st.number_input("أدخل عدد الصفوف:", min_value=1, max_value=10, value=3)
            columns = st.number_input("أدخل عدد الأعمدة:", min_value=2, max_value=10, value=4)
            
            if columns <= rows:
                st.error("عدد الأعمدة يجب أن يكون أكبر من عدد الصفوف في نظام المعادلات")
                st.stop()
                
        except Exception as e:
            st.error(f"خطأ في إدخال الأبعاد: {str(e)}")
            st.stop()

        st.write("أدخل قيم المصفوفة:")
        matrix = []
        try:
            for i in range(int(rows)):
                row = []
                cols = st.columns(int(columns))
                for j in range(int(columns)):
                    with cols[j]:
                        val = st.number_input(f"الصف {i + 1}, العمود {j + 1}", value=0.0, step=1.0, key=f"input_{i}_{j}")
                        row.append(val)
                matrix.append(row)
        except Exception as e:
            st.error(f"خطأ في إدخال قيم المصفوفة: {str(e)}")
            st.stop()

        if st.button("حل"):
            solution_text = solve_matrix(matrix)
            st.subheader("الناتج:")
            if isinstance(solution_text, str):
                if "خطأ" in solution_text:
                    st.error(solution_text)
                else:
                    st.text(solution_text)
            else:
                st.text(solution_text)

    # صفحة جمع المصفوفات
    elif page == "جمع المصفوفات":
        st.title("جمع المصفوفات")
        
        try:
            rows = st.number_input("أدخل عدد الصفوف:", min_value=1, max_value=10, value=3)
            columns = st.number_input("أدخل عدد الأعمدة:", min_value=1, max_value=10, value=3)
        except Exception as e:
            st.error(f"خطأ في إدخال الأبعاد: {str(e)}")
            st.stop()

        try:
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
        except Exception as e:
            st.error(f"خطأ في إدخال قيم المصفوفات: {str(e)}")
            st.stop()

        if st.button("جمع المصفوفات"):
            result = add_matrices(matrix1, matrix2)
            st.subheader("الناتج:")
            if isinstance(result, str):
                st.error(result)
            else:
                st.text(format_matrix(result))

    # صفحة طرح المصفوفات
    elif page == "طرح المصفوفات":
        st.title("طرح المصفوفات")
        
        try:
            rows = st.number_input("أدخل عدد الصفوف:", min_value=1, max_value=10, value=3)
            columns = st.number_input("أدخل عدد الأعمدة:", min_value=1, max_value=10, value=3)
        except Exception as e:
            st.error(f"خطأ في إدخال الأبعاد: {str(e)}")
            st.stop()

        try:
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
        except Exception as e:
            st.error(f"خطأ في إدخال قيم المصفوفات: {str(e)}")
            st.stop()

        if st.button("طرح المصفوفات"):
            result = subtract_matrices(matrix1, matrix2)
            st.subheader("الناتج:")
            if isinstance(result, str):
                st.error(result)
            else:
                st.text(format_matrix(result))

    # صفحة ضرب المصفوفات
    elif page == "ضرب المصفوفات":
        st.title("ضرب المصفوفات")
        
        try:
            rows1 = st.number_input("أدخل عدد الصفوف للمصفوفة الأولى:", min_value=1, max_value=10, value=3)
            columns1 = st.number_input("أدخل عدد الأعمدة للمصفوفة الأولى:", min_value=1, max_value=10, value=3)
            rows2 = columns1  # عدد صفوف المصفوفة الثانية يجب أن يساوي عدد أعمدة المصفوفة الأولى
            columns2 = st.number_input("أدخل عدد الأعمدة للمصفوفة الثانية:", min_value=1, max_value=10, value=3)
            
            if columns1 != rows2:
                st.error("عدد أعمدة المصفوفة الأولى يجب أن يساوي عدد صفوف المصفوفة الثانية")
                st.stop()
                
        except Exception as e:
            st.error(f"خطأ في إدخال الأبعاد: {str(e)}")
            st.stop()

        try:
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
        except Exception as e:
            st.error(f"خطأ في إدخال قيم المصفوفات: {str(e)}")
            st.stop()

        if st.button("ضرب المصفوفات"):
            result = multiply_matrices(matrix1, matrix2)
            st.subheader("الناتج:")
            if isinstance(result, str):
                st.error(result)
            else:
                st.text(format_matrix(result))
                
    # صفحة ترانسبوز المصفوفة
    elif page == "ترانسبوز المصفوفة":
        st.title("ترانسبوز المصفوفة")
        
        try:
            rows = st.number_input("أدخل عدد الصفوف:", min_value=1, max_value=10, value=3)
            columns = st.number_input("أدخل عدد الأعمدة:", min_value=1, max_value=10, value=3)
        except Exception as e:
            st.error(f"خطأ في إدخال الأبعاد: {str(e)}")
            st.stop()

        try:
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
        except Exception as e:
            st.error(f"خطأ في إدخال قيم المصفوفة: {str(e)}")
            st.stop()

        if st.button("ترانسبوز المصفوفة"):
            result = transpose_matrix(matrix)
            st.subheader("الناتج:")
            if isinstance(result, str):
                st.error(result)
            else:
                st.text(format_matrix(result))
            
    # صفحة عكس المصفوفة
    elif page == "عكس المصفوفة":
        st.title("عكس المصفوفة باستخدام Gauss-Jordan Elimination")
        
        try:
            rows = st.number_input("أدخل عدد الصفوف:", min_value=1, max_value=10, value=3)
            columns = st.number_input("أدخل عدد الأعمدة:", min_value=1, max_value=10, value=3)
            
            if rows != columns:
                st.error("يجب أن تكون المصفوفة مربعة (عدد الصفوف = عدد الأعمدة)")
                st.stop()
                
        except Exception as e:
            st.error(f"خطأ في إدخال الأبعاد: {str(e)}")
            st.stop()

        try:
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
        except Exception as e:
            st.error(f"خطأ في إدخال قيم المصفوفة: {str(e)}")
            st.stop()

        if st.button("عكس المصفوفة"):
            solution_text, inverse_matrix = invert_matrix(matrix)
            st.subheader("الناتج:")
            if inverse_matrix is not None:
                st.text(solution_text)
                st.text("المصفوفة المعكوسة:")
                st.text(format_matrix(inverse_matrix))
            else:
                st.error(solution_text)
except Exception as e:
    st.error(f"حدث خطأ غير متوقع: {str(e)}")
