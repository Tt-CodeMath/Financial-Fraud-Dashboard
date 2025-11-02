import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import plotly.express as px
import plotly.graph_objects as go
import gc
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression

st.set_page_config(page_title="Fraud Detection Dashboard", layout="wide")
st.title("🔒 Fraud Detection Dashboard")

# --- Tabs ---
tab1, tab2, tab3 = st.tabs(["📁 Dữ liệu & Phân tích", "📈 Đánh giá mô hình", "🤖 Dự đoán"])

# --- Shared ---
fraud_col = None

# ------------------- TAB 1: DỮ LIỆU ------------------- #
with tab1:
    st.subheader("📊 Phân bố thời gian giao dịch (`step`) ")

    @st.cache_data
    def load_data():
        return pd.read_csv(r"D:\FPT_Material\Financial-Fraud-Dashboard\data\PS_20174392719_1491204439457_log.csv")

    df = load_data()
    # Tạo bản sao để tránh ảnh hưởng df gốc
    prep_df = df.copy()

    # Tạo đặc trưng: Giờ trong ngày (0–23)
    prep_df['hour_of_day'] = prep_df['step'] % 24

    # Ngưỡng step được xem là nguy hiểm
    RISK_THRESHOLD_STEP = 400
    prep_df['is_high_risk_step_period'] = (prep_df['step'] > RISK_THRESHOLD_STEP).astype(int)

    # Tạo label dễ hiểu cho biểu đồ
    prep_df["risk_period_label"] = prep_df["is_high_risk_step_period"].map({
        0: f"Normal Period (step ≤ {RISK_THRESHOLD_STEP})",
        1: f"High-Risk Period (step > {RISK_THRESHOLD_STEP})"
    })

    # _____________________________________________
    
    fig = px.histogram(
        df,
        x='step',
        nbins=100,
        title="Distribution of Time Step (step)",
        color_discrete_sequence=['dodgerblue']
    )
    fig.update_layout(
        xaxis_title="Step (Time Step)",
        yaxis_title="Number of Transactions",
        bargap=0.01,
        template="plotly_white"
    )

    st.plotly_chart(fig, use_container_width=True)
    # _____________________________________________
    
        # Tính toán tổng và gian lận theo step
    step_analysis = df.groupby("step").agg(
        total_transactions=('isFraud', 'count'),
        fraud_transactions=('isFraud', 'sum')
    ).reset_index()

    step_analysis['fraud_percentage'] = (
        step_analysis['fraud_transactions'] / step_analysis['total_transactions'] * 100
    )

    # Vẽ biểu đồ dual-axis
    fig2 = go.Figure()

    # Bar: Tổng số giao dịch
    fig2.add_trace(go.Bar(
        x=step_analysis["step"],
        y=step_analysis["total_transactions"],
        name="Total Transactions",
        marker_color="lightblue",
        yaxis="y1"
    ))

    # Line: % gian lận
    fig2.add_trace(go.Scatter(
        x=step_analysis["step"],
        y=step_analysis["fraud_percentage"],
        name="Fraud Percentage (%)",
        mode="lines+markers",
        marker=dict(color="red"),
        yaxis="y2"
    ))

    # Cấu hình layout 2 trục
    fig2.update_layout(
        title="📈 Total Transactions and Fraud Percentage by Step",
        xaxis=dict(title="Step (Time Step)"),
        yaxis=dict(title="Total Transactions", side="left", showgrid=False),
        yaxis2=dict(
            title="Fraud Percentage (%)",
            overlaying="y",
            side="right",
            showgrid=False
        ),
        legend=dict(x=0.01, y=0.99),
        bargap=0.05,
        template="plotly_white"
    )

    # Hiển thị biểu đồ trong Streamlit
    st.subheader("📈 Giao dịch và tỷ lệ gian lận theo thời gian (`step`)")
    st.plotly_chart(fig2, use_container_width=True)
    # _____________________________________________
        # Lọc chỉ các giao dịch gian lận
    fraud_df = df[df['isFraud'] == 1]

    # Plotly histogram chỉ cho isFraud = 1
    fig3 = px.histogram(
        fraud_df,
        x='step',
        nbins=100,
        title="Distribution of Time Step (step) for Fraudulent Transactions",
        color_discrete_sequence=['orangered']
    )

    fig3.update_layout(
        xaxis_title="Step (Time Step)",
        yaxis_title="Number of Fraudulent Transactions",
        bargap=0.01,
        template="plotly_white"
    )

    # Hiển thị biểu đồ
    st.subheader("🔍 Phân bố thời gian giao dịch gian lận (`isFraud = 1`)")
    st.plotly_chart(fig3, use_container_width=True)
    # _____________________________________________
    

    # Biểu đồ cột nhóm: số lượng giao dịch theo giờ, phân chia theo isFraud
    fig4 = px.histogram(
        prep_df,
        x="hour_of_day",
        color="isFraud",
        barmode="group",
        color_discrete_map={0: '#00B7EB', 1: '#FF1493'},
        category_orders={"hour_of_day": list(range(24))},
        title="Fraud vs. Normal Transactions by Hour of the Day"
    )

    fig4.update_layout(
        xaxis_title="Hour of the Day (0–23)",
        yaxis_title="Number of Transactions",
        legend_title="Fraud Status",
        legend=dict(x=0.8),
        template="plotly_white"
    )

    fig4.update_traces(marker_line_width=1)
    st.subheader("🕒 Phân bố gian lận theo giờ trong ngày")
    st.plotly_chart(fig4, use_container_width=True)

    # _____________________________________________
    # Biểu đồ cột nhóm: gian lận theo thời kỳ nguy hiểm
    # Tổng hợp trước: đếm số giao dịch theo loại thời gian & gian lận
    agg_df = prep_df.groupby(["risk_period_label", "isFraud"]).size().reset_index(name="count")

    # Tách dữ liệu thành fraud / non-fraud
    normal_counts = agg_df[agg_df['isFraud'] == 0]
    fraud_counts = agg_df[agg_df['isFraud'] == 1]

    # Vẽ bar chart thủ công
    fig5 = go.Figure()

    fig5.add_trace(go.Bar(
        x=normal_counts["risk_period_label"],
        y=normal_counts["count"],
        name="Normal",
        marker_color="#00B7EB"
    ))

    fig5.add_trace(go.Bar(
        x=fraud_counts["risk_period_label"],
        y=fraud_counts["count"],
        name="Fraud",
        marker_color="#FF1493"
    ))

    fig5.update_layout(
        barmode='group',
        title="Fraud vs. Normal Transactions by High-Risk Period",
        xaxis_title="Time Period",
        yaxis_title="Number of Transactions",
        template="plotly_white"
    )

    st.subheader("⚠️ Phân bố gian lận theo vùng thời gian nguy hiểm (`step > 400`)")
    st.plotly_chart(fig5, use_container_width=True)

# --- CLEAN UP: Giải phóng bộ nhớ sau tab1 ---
# Xóa các biến không còn sử dụng
del df, prep_df, fraud_df, step_analysis, agg_df

# Xóa tất cả các figure plotly
del fig, fig2, fig3, fig4, fig5

# Nếu dùng cache thì xóa cache để giải phóng RAM
load_data.clear()

# Ép Python thu hồi bộ nhớ
gc.collect()
    
# ------------------- TAB 2: ĐÁNH GIÁ MÔ HÌNH ------------------- #
with tab2:
    st.subheader("📊 Đánh giá mô hình trên dữ liệu kiểm thử")

    st.sidebar.header("Tải mô hình đã huấn luyện và dữ liệu test")
    eval_model_file = st.sidebar.file_uploader("Upload mô hình (pkl/joblib)", type=["pkl", "joblib"], key="eval_model")
    test_data_file = st.sidebar.file_uploader("Upload file test (CSV, có cột isFraud)", type=["csv"], key="test_data")

    if eval_model_file and test_data_file:
        try:
            # Load model và test data
            model = joblib.load(eval_model_file)
            test_df = pd.read_csv(test_data_file)

            if 'isFraud' not in test_df.columns:
                st.error("⚠️ File test phải chứa cột `isFraud`.")
            else:
                # ⚠️ Chỉ giữ lại các cột đã dùng trong huấn luyện
                feature_names = [
                    "step", "amount", "isFlaggedFraud",
                    "errorBalanceOrig", "errorBalanceDest", "emptiedAccountOrig",
                    "hour_of_day", "is_high_risk_step_period",
                    "type_CASH_IN", "type_CASH_OUT", "type_DEBIT", "type_PAYMENT", "type_TRANSFER"
                ] 

                # Kiểm tra thiếu cột nào không
                missing_cols = [col for col in feature_names if col not in test_df.columns]
                if missing_cols:
                    st.error(f"⚠️ File test thiếu các cột sau: {', '.join(missing_cols)}")
                else:
                    X_test = test_df[feature_names]
                    y_test = test_df["isFraud"]

                    # Dự đoán
                    y_pred = model.predict(X_test)

                    # Confusion Matrix
                    st.subheader("📉 Confusion Matrix")
                    cm = confusion_matrix(y_test, y_pred)
                    fig_cm = px.imshow(
                        cm,
                        text_auto=True,
                        color_continuous_scale='blues',
                        x=["Predicted: 0", "Predicted: 1"],
                        y=["Actual: 0", "Actual: 1"],
                        labels=dict(x="Predicted", y="Actual", color="Count"),
                        title="Confusion Matrix"
                    )
                    st.plotly_chart(fig_cm, use_container_width=True)

                    # Classification Report
                    st.subheader("📑 Classification Report")
                    report_str = classification_report(y_test, y_pred, output_dict=False)
                    st.code(report_str)

                    # Hiển thị các chỉ số tổng hợp
                    st.subheader("📌 Các chỉ số đánh giá")
                    report_dict = classification_report(y_test, y_pred, output_dict=True)
                    st.metric("Precision (Fraud)", f"{report_dict['1']['precision']:.4f}")
                    st.metric("Recall (Fraud)", f"{report_dict['1']['recall']:.4f}")
                    st.metric("F1-score (Fraud)", f"{report_dict['1']['f1-score']:.4f}")
                    st.metric("Accuracy", f"{accuracy_score(y_test, y_pred):.4f}")

                    # ---------------- Feature Importance ----------------
                    st.subheader("📌 Feature Importance (Logistic Regression Coefficients)")
                    try:
                        classifier = model.named_steps['classifier']
                        feature_names_used = X_test.columns
                        importances = classifier.coef_[0]

                        feature_importance_df = pd.DataFrame({
                            'Feature': feature_names_used,
                            'Importance': importances
                        }).sort_values(by='Importance', ascending=False)

                        fig_imp, ax = plt.subplots(figsize=(10, 8))
                        sns.barplot(
                            x='Importance',
                            y='Feature',
                            data=feature_importance_df,
                            palette='viridis',
                            ax=ax
                        )
                        ax.set_title('Feature Importance (Logistic Regression)', fontsize=14)
                        st.pyplot(fig_imp)
                    except Exception as e:
                        st.error(f"❌ Không thể hiển thị Feature Importance: {e}")

                    # ---------------- SHAP Explanation ----------------
                    st.subheader("🔍 Giải thích mô hình với SHAP")
                    try:
                        import shap
                        explainer = shap.LinearExplainer(classifier, X_test, feature_perturbation="interventional")
                        shap_values = explainer.shap_values(X_test)

                        # Giao dịch gian lận
                        fraud_idx = y_test[y_test == 1].index[0]
                        st.markdown(f"**📌 Giao dịch gian lận - Index: {fraud_idx}**")
                        
                        fig_fraud, ax_fraud = plt.subplots(figsize=(10, 6))
                        shap.plots._waterfall.waterfall_legacy(
                            expected_value=explainer.expected_value,
                            shap_values=shap_values[fraud_idx],
                            features=X_test.loc[fraud_idx],
                            feature_names=X_test.columns.tolist(),
                            max_display=10,
                            show=False
                        )
                        st.pyplot(fig_fraud)

                        # Giao dịch không gian lận
                        nonfraud_idx = y_test[y_test == 0].index[0]
                        st.markdown(f"**📌 Giao dịch không gian lận - Index: {nonfraud_idx}**")
                        
                        fig_nonfraud, ax_nonfraud = plt.subplots(figsize=(10, 6))
                        shap.plots._waterfall.waterfall_legacy(
                            expected_value=explainer.expected_value,
                            shap_values=shap_values[nonfraud_idx],
                            features=X_test.loc[nonfraud_idx],
                            feature_names=X_test.columns.tolist(),
                            max_display=10,
                            show=False
                        )
                        st.pyplot(fig_nonfraud)

                    except Exception as e:
                        st.error(f"❌ Không thể tạo biểu đồ SHAP: {e}")
        except Exception as e:
            st.error(f"❌ Đã xảy ra lỗi khi đánh giá mô hình: {e}")
    else:
        st.info("📤 Vui lòng tải mô hình và dữ liệu test để đánh giá.")

# ------------------- TAB 3: DỰ ĐOÁN ------------------- #
with tab3:
    st.sidebar.header("Upload mô hình đã huấn luyện")
    model_file = st.sidebar.file_uploader("Upload mô hình (pkl/joblib)", type=["pkl", "joblib"])

    if model_file:
        model = joblib.load(model_file)
        st.subheader("🔎 Dự đoán một giao dịch")
        st.markdown("*Nhập giá trị đặc trưng, cách nhau bằng dấu phẩy.*")

        feature_names = [
            "step", "amount", "isFlaggedFraud", "errorBalanceOrig", "errorBalanceDest",
            "emptiedAccountOrig", "hour_of_day", "is_high_risk_step_period",
            "type_CASH_IN", "type_CASH_OUT", "type_DEBIT", "type_PAYMENT", "type_TRANSFER"
        ]
        st.caption("**Cột đầu vào:** " + ", ".join(feature_names))
        input_str = st.text_input("Nhập giao dịch:")

        if st.button("Dự đoán"):
            try:
                values = [float(x.strip()) for x in input_str.split(",")]
                pred = model.predict([values])[0]
                st.success(f"👉 Kết quả: {'Gian lận (Fraud)' if pred == 1 else 'Không gian lận'}")
            except:
                st.error("⚠️ Dữ liệu không hợp lệ. Hãy nhập đúng số lượng và định dạng.")

        # Batch prediction
        st.subheader("📥 Dự đoán hàng loạt")
        batch_file = st.file_uploader("Upload file để dự đoán hàng loạt (không có cột Prediction)", type=["csv"], key="batch")

        if batch_file:
            batch_df = pd.read_csv(batch_file)
            try:
                preds = model.predict(batch_df)
                batch_df['Prediction'] = preds
                st.success("✅ Dự đoán hoàn tất!")
                st.dataframe(batch_df.head())
                csv = batch_df.to_csv(index=False)
                st.download_button("📥 Tải kết quả", csv, file_name="predicted_results.csv")
            except Exception as e:
                st.error(f"Lỗi khi dự đoán: {e}")
