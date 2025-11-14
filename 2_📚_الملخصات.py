import streamlit as st
import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma

# --- إعدادات الصفحة والاتصال ---
st.set_page_config(page_title="ملخصات ذكية", page_icon="📚")
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    st.error("❌ خطأ في الاتصال.")
    st.stop()

DB_DIR = "persistent_db"

# --- تحميل الخزينة العلمية ---
@st.cache_resource
def load_knowledge_base():
    if not os.path.exists(DB_DIR):
        return None 
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key)
    vector_store = Chroma(persist_directory=DB_DIR, embedding_function=embeddings)
    return vector_store

vector_store = load_knowledge_base()

# --- واجهة المستخدم ---
st.title("📚 خدمة الملخصات الأكاديمية")
st.caption("يعمل هذا النظام بناءً على الخزينة العلمية (كتبك ومراجعك).")

if vector_store:
    st.subheader("أدخل الموضوع أو الفصل الذي تريد تلخيصه:")
    topic = st.text_input("مثال: لخص لي أهم النقاط في 'الفصل الخامس من كتاب...'")
    
    if st.button("🚀 ابدأ التلخيص", type="primary"):
        if not topic:
            st.warning("الرجاء إدخال الموضوع أولاً.")
        else:
            with st.spinner("جارٍ قراءة المراجع واستخلاص الملخص..."):
                try:
                    docs = vector_store.similarity_search(topic, k=15) 
                    context_text = "\n\n".join([doc.page_content for doc in docs])
                    
                    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=api_key, temperature=0.2)
                    
                    prompt = f"""
                    أنت خبير في تلخيص المواد الأكاديمية المعقدة.
                    مهمتك هي إنشاء ملخص شامل وواضح للموضوع التالي: "{topic}"
                    
                    استخدم المعلومات التالية من الخزينة العلمية:
                    [المعلومات المسترجعة]:
                    {context_text}
                    
                    تعليمات صارمة (نسخة v2 - ضد النسخ):
                    1. قم بإنشاء ملخص "على شكل نقاط" (bullet points) يغطي الأفكار الرئيسية.
                    2. يجب أن يكون الملخص دقيقاً ومباشراً وسهل الفهم.
                    3. [الأمر الأهم]: يجب عليك إعادة صياغة كل المعلومات بأسلوبك الأكاديمي الخاص. ممنوع منعاً باتاً نسخ أي جملة حرفياً (No direct quotes) من المعلومات المسترجعة.
                    4. لا تذكر "المصدر" أو "المعلومات". قدم الملخص كحقائق.
                    """
                    
                    response = llm.invoke(prompt)
                    
                    st.markdown("### 📝 الملخص الرئيسي:")
                    st.success("تم إنشاء الملخص بنجاح!")
                    st.markdown(response.content)
                            
                except Exception as e:
                    st.error(f"حدث خطأ أثناء التوليد: {e}")
else:
    st.error("⚠️ لم يتم العثور على الخزينة العلمية. يرجى تشغيل `build_database.py` أولاً.")

# --- التذييل المحدث ---
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #808080; font-size: 14px;'>
    الذكاء الاصطناعي لخدمة البحث العلمي 📚 | 2025
    </div>
    """, 
    unsafe_allow_html=True
)