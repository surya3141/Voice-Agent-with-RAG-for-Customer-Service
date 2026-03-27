"""TechPulse Voice Agent Monitoring Dashboard.

Streamlit-based dashboard for monitoring the voice agent system,
simulating calls, analyzing costs, and browsing the knowledge base.
"""

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import streamlit as st

# Ensure the project root is on the Python path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import get_config, DATA_DIR
from src.monitoring.cost_tracker import CostTracker
from src.rag.pipeline import RAGPipeline
from src.rag.vector_store import KnowledgeStore, VectorStoreError
from src.security.rbac import RBACManager, Role, User

# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="TechPulse Voice Agent Dashboard",
    page_icon="🎙️",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Predefined user credentials (demo purposes)
# ---------------------------------------------------------------------------
_DEMO_USERS: dict[str, dict[str, str]] = {
    "admin": {"password": "admin123", "role": "admin"},
    "support": {"password": "support123", "role": "support"},
    "viewer": {"password": "viewer123", "role": "viewer"},
}

_ROLE_MAP: dict[str, Role] = {
    "admin": Role.ADMIN,
    "support": Role.SUPPORT,
    "viewer": Role.VIEWER,
}


# ---------------------------------------------------------------------------
# Session-state helpers
# ---------------------------------------------------------------------------
def _init_session_state() -> None:
    """Initialise default session-state values."""
    defaults: dict[str, object] = {
        "authenticated": False,
        "username": "",
        "role": "",
        "token": "",
        "call_log": [],
        "cost_tracker": CostTracker(alert_threshold=50.0),
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


_init_session_state()


# ---------------------------------------------------------------------------
# Cached resource loaders
# ---------------------------------------------------------------------------
@st.cache_resource
def load_knowledge_store() -> KnowledgeStore | None:
    """Load and return the KnowledgeStore with FAQ data."""
    try:
        store = KnowledgeStore()
        faqs_path = str(DATA_DIR / "faqs.json")
        store.load_faqs(faqs_path)
        orders_path = str(DATA_DIR / "orders.json")
        store.load_orders(orders_path)
        appointments_path = str(DATA_DIR / "appointments.json")
        store.load_appointments(appointments_path)
        return store
    except VectorStoreError as exc:
        st.error(f"Failed to load knowledge store: {exc}")
        return None
    except FileNotFoundError as exc:
        st.error(f"Data file not found: {exc}")
        return None


@st.cache_resource
def get_rag_pipeline() -> RAGPipeline | None:
    """Build the RAG pipeline (requires a valid Groq API key)."""
    try:
        config = get_config()
        if not config.groq.api_key:
            return None

        from src.llm.groq_llm import GroqLLM

        store = load_knowledge_store()
        if store is None:
            return None
        llm = GroqLLM(config.groq)
        return RAGPipeline(knowledge_store=store, llm=llm)
    except Exception as exc:  # noqa: BLE001 – broad but intentional
        st.error(f"Failed to initialise RAG pipeline: {exc}")
        return None


def _get_rbac_manager() -> RBACManager:
    """Return an RBAC manager using the application secret key."""
    config = get_config()
    secret = config.security.admin_secret_key or "default-dashboard-secret"
    return RBACManager(secret_key=secret)


# ---------------------------------------------------------------------------
# Authentication sidebar
# ---------------------------------------------------------------------------
def _render_sidebar() -> str:
    """Render sidebar with login form and navigation. Return selected page."""
    with st.sidebar:
        st.title("🎙️ TechPulse Agent")
        st.divider()

        if not st.session_state["authenticated"]:
            st.subheader("Login")
            username = st.text_input("Username", key="login_username")
            password = st.text_input("Password", type="password", key="login_password")

            if st.button("Login", use_container_width=True):
                if username in _DEMO_USERS and _DEMO_USERS[username]["password"] == password:
                    role_str = _DEMO_USERS[username]["role"]
                    role = _ROLE_MAP[role_str]
                    user = User(
                        username=username,
                        role=role,
                        created_at=datetime.now(timezone.utc),
                    )
                    rbac = _get_rbac_manager()
                    token = rbac.create_token(user)

                    st.session_state["authenticated"] = True
                    st.session_state["username"] = username
                    st.session_state["role"] = role_str
                    st.session_state["token"] = token
                    st.rerun()
                else:
                    st.error("Invalid credentials")

            return "Overview"

        # Authenticated user info
        st.success(f"Logged in as **{st.session_state['username']}**")
        st.caption(f"Role: `{st.session_state['role'].upper()}`")

        if st.button("Logout", use_container_width=True):
            for key in ("authenticated", "username", "role", "token"):
                st.session_state[key] = "" if key != "authenticated" else False
            st.rerun()

        st.divider()

        # Navigation – available pages depend on role
        pages = ["Overview"]
        role = st.session_state["role"]
        if role in ("support", "admin"):
            pages.append("Call Simulator")
        if role == "admin":
            pages.append("Cost Analysis")
        pages.append("Knowledge Base")

        selected = st.radio("Navigation", pages, label_visibility="collapsed")
        return selected  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Page: Overview
# ---------------------------------------------------------------------------
def _page_overview() -> None:
    """Render the Overview page with key metrics and call log."""
    st.header("📊 Dashboard Overview")

    call_log: list[dict[str, object]] = st.session_state["call_log"]
    tracker: CostTracker = st.session_state["cost_tracker"]

    total_calls = len(call_log)
    avg_confidence = (
        sum(float(c.get("confidence", 0)) for c in call_log) / total_calls
        if total_calls
        else 0.0
    )
    total_cost = tracker.get_total_cost()
    active_appointments = sum(
        1 for c in call_log if c.get("intent") == "appointment"
    )

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Calls", total_calls)
    col2.metric("Avg Confidence", f"{avg_confidence:.2f}")
    col3.metric("Total Cost", f"${total_cost:.4f}")
    col4.metric("Active Appointments", active_appointments)

    st.subheader("Recent Call Log")
    if call_log:
        st.dataframe(call_log, use_container_width=True)
    else:
        st.info("No calls recorded yet. Use the Call Simulator to generate entries.")


# ---------------------------------------------------------------------------
# Page: Call Simulator
# ---------------------------------------------------------------------------
def _page_call_simulator() -> None:
    """Render the Call Simulator page."""
    st.header("📞 Call Simulator")

    pipeline = get_rag_pipeline()
    if pipeline is None:
        st.info(
            "RAG pipeline is unavailable. Please ensure a valid GROQ_API_KEY "
            "is set in your environment."
        )
        return

    query = st.text_input("Enter a customer query:", key="sim_query")

    if st.button("Process Query", use_container_width=True) and query:
        with st.spinner("Processing…"):
            try:
                result = pipeline.query(query)
                st.subheader("Response")
                st.write(result.get("answer", ""))

                col1, col2 = st.columns(2)
                with col1:
                    st.markdown(f"**Detected Intent:** `{result.get('intent', 'N/A')}`")
                with col2:
                    sources = result.get("sources", [])
                    st.markdown(f"**Sources:** {len(sources)} documents retrieved")

                if sources:
                    with st.expander("View Sources"):
                        for i, src in enumerate(sources, 1):
                            st.markdown(f"**Source {i}**")
                            st.write(src.get("document", ""))
                            st.caption(f"Metadata: {src.get('metadata', {})}")

                # Record in session call log
                tracker: CostTracker = st.session_state["cost_tracker"]
                tracker.record_llm_usage(input_tokens=150, output_tokens=80)

                entry: dict[str, object] = {
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "query": query,
                    "intent": result.get("intent", ""),
                    "confidence": 0.95,
                    "response_preview": str(result.get("answer", ""))[:100],
                }
                st.session_state["call_log"].append(entry)
            except Exception as exc:  # noqa: BLE001
                st.error(f"Error processing query: {exc}")


# ---------------------------------------------------------------------------
# Page: Cost Analysis
# ---------------------------------------------------------------------------
def _page_cost_analysis() -> None:
    """Render the Cost Analysis page (admin only)."""
    st.header("💰 Cost Analysis")

    tracker: CostTracker = st.session_state["cost_tracker"]

    # Cost breakdown
    st.subheader("Cost Breakdown by Service")
    breakdown = tracker.get_cost_breakdown()
    if breakdown:
        cols = st.columns(len(breakdown))
        for col, (service, cost) in zip(cols, breakdown.items()):
            col.metric(service.title(), f"${cost:.4f}")
    else:
        st.info("No cost data recorded yet.")

    st.divider()

    # ROI Report
    st.subheader("ROI Report")
    calls_handled = len(st.session_state["call_log"]) or 1
    roi = tracker.generate_roi_report(calls_handled=calls_handled)

    r1, r2, r3 = st.columns(3)
    r1.metric("Total AI Cost", f"${roi['total_cost']:.4f}")
    r2.metric("Human Equivalent Cost", f"${roi['human_equivalent_cost']:.2f}")
    r3.metric("ROI", f"{roi['roi_percentage']:.1f}%")

    s1, s2 = st.columns(2)
    s1.metric("Cost Per Call", f"${roi['cost_per_call']:.4f}")
    s2.metric("Total Savings", f"${roi['savings']:.2f}")

    st.divider()

    # Alerts
    st.subheader("Alerts")
    alerts = tracker.check_alerts()
    if alerts:
        for alert in alerts:
            st.warning(alert)
    else:
        st.success("All costs within acceptable thresholds.")


# ---------------------------------------------------------------------------
# Page: Knowledge Base
# ---------------------------------------------------------------------------
def _page_knowledge_base() -> None:
    """Render the Knowledge Base page with searchable FAQs."""
    st.header("📚 Knowledge Base")

    faqs_path = DATA_DIR / "faqs.json"
    try:
        with open(faqs_path) as f:
            faqs: list[dict[str, str]] = json.load(f)
    except FileNotFoundError:
        st.error(f"FAQs file not found at {faqs_path}")
        return
    except json.JSONDecodeError as exc:
        st.error(f"Invalid JSON in FAQs file: {exc}")
        return

    search_term = st.text_input("🔍 Search FAQs", key="faq_search")

    if search_term:
        term_lower = search_term.lower()
        faqs = [
            faq
            for faq in faqs
            if term_lower in faq.get("question", "").lower()
            or term_lower in faq.get("answer", "").lower()
            or term_lower in faq.get("category", "").lower()
        ]

    if not faqs:
        st.info("No FAQs match your search.")
        return

    st.caption(f"Showing {len(faqs)} FAQ(s)")

    for faq in faqs:
        with st.expander(f"**{faq.get('question', 'Untitled')}**  — _{faq.get('category', '')}_"):
            st.write(faq.get("answer", ""))
            st.caption(f"ID: {faq.get('id', 'N/A')}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    """Entry-point for the Streamlit dashboard."""
    page = _render_sidebar()

    if not st.session_state["authenticated"]:
        st.title("🎙️ TechPulse Voice Agent Dashboard")
        st.info("Please log in using the sidebar to access the dashboard.")
        return

    if page == "Overview":
        _page_overview()
    elif page == "Call Simulator":
        _page_call_simulator()
    elif page == "Cost Analysis":
        _page_cost_analysis()
    elif page == "Knowledge Base":
        _page_knowledge_base()


if __name__ == "__main__":
    main()
