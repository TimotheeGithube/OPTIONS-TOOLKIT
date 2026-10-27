"""
Options Toolkit - Portfolio | Arbitrage | Delta/Delta-Gamma Hedge | SABR (+ Monte Carlo)
Run:  pip install streamlit numpy scipy pandas matplotlib sympy yfinance openpyxl
      streamlit run options_toolkit.py
"""
import io
import numpy as np, pandas as pd, sympy as sp, streamlit as st
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from scipy.stats import norm
from scipy.optimize import least_squares, brentq

st.set_page_config(page_title="Options Toolkit: Midterm Exam Timothée Macal", page_icon="", layout="wide")
import streamlit.components.v1 as components
components.html("""<script>
const d = window.parent.document;
d.documentElement.setAttribute('lang', 'en'); d.documentElement.setAttribute('translate', 'no');
const m = d.createElement('meta'); m.name = 'google'; m.content = 'notranslate'; d.head.appendChild(m);
</script>""", height=0)
N, n_pdf = norm.cdf, norm.pdf
from cycler import cycler
plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.alpha": .25,
                     "axes.prop_cycle": cycler(color=["#3a7bd5", "#f28e2b", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]),
                     "figure.autolayout": True, "axes.titleweight": "bold"})

# ----------------------------------------------------------------- core maths
def bs(S, K, r, sig, T, cp="call"):
    """Black-Scholes (no dividends) -> price, delta, gamma."""
    if T <= 1e-9 or sig <= 1e-9:
        if cp == "call": return max(S-K, 0.0), float(S > K), 0.0
        return max(K-S, 0.0), -float(S < K), 0.0
    d1 = (np.log(S/K)+(r+.5*sig**2)*T)/(sig*np.sqrt(T)); d2 = d1-sig*np.sqrt(T)
    if cp == "call": p, dl = S*N(d1)-K*np.exp(-r*T)*N(d2), N(d1)
    else:            p, dl = K*np.exp(-r*T)*N(-d2)-S*N(-d1), N(d1)-1
    return float(p), float(dl), float(n_pdf(d1)/(S*sig*np.sqrt(T)))

def replicate(S, K, r, sig, T, cp):
    """Synthetic option = shares + bond. Returns (shares, bond cash-flow today)."""
    d1 = (np.log(S/K)+(r+.5*sig**2)*T)/(sig*np.sqrt(T)); d2 = d1-sig*np.sqrt(T)
    if cp == "call": return N(d1), -K*np.exp(-r*T)*N(d2)
    return -N(-d1), K*np.exp(-r*T)*N(-d2)

def arb_steps(S, K, r, sig, T, cp, mkt, tol=1e-4):
    sh, bond = replicate(S, K, r, sig, T, cp); syn = sh*S+bond
    out = [f"Synthetic {cp} (stock + bond) costs **{syn:.4f}**; market price is **{mkt:.4f}**."]
    if abs(mkt-syn) < tol: return out+["No arbitrage: option is fairly priced."], 0.0
    if abs(sh) < 1e-4 and abs(bond) < 1e-4:
        return out+[f"The option is worth ~0 under the model (deep out-of-the-money for this S, K, T, sigma). "
                    f"{'Selling it at %.4f is the arbitrage: no hedge is needed.' % mkt if mkt > syn else 'Nothing to exploit.'}"], abs(mkt-syn)
    over = mkt > syn
    out.append(f"The option is **{'OVER' if over else 'UNDER'}-priced** -> arbitrage profit today = **{abs(mkt-syn):.4f}** per option.")
    sgn = 1 if over else -1                      # +1: sell option, buy synthetic
    steps = [f"{'Sell' if over else 'Buy'} 1 {cp} (K={K}, T={T}) at {mkt:.4f}.",
             f"{'Buy' if sgn*sh > 0 else 'Short-sell'} {abs(sh):.4f} shares at {S:.4f} (cash {-sgn*sh*S:+.4f}).",
             (f"{'Lend (buy zero-coupon bond worth' if sgn*bond > 0 else 'Borrow (issue bond worth'} {abs(bond):.4f}) at r={r:.2%}"
              f" (cash {-sgn*bond:+.4f})."),
             f"Net cash today = {abs(mkt-syn):.4f} > 0. Rebalance delta as S moves (dynamic replication); at T the synthetic exactly offsets the option payoff.",
             f"At maturity: unwind stock and bond; the option payoff is cancelled, you keep the profit."]
    return out+[f"{i+1}. {s}" for i, s in enumerate(steps)], abs(mkt-syn)

def hagan(F, K, T, a, b, rho, nu):
    """Hagan et al. (2002) lognormal SABR implied vol (vectorised in K)."""
    K = np.asarray(K, float); lf = np.log(F/K); fk = (F*K)**((1-b)/2)
    z = nu/a*fk*lf; x = np.log((np.sqrt(1-2*rho*z+z*z)+z-rho)/(1-rho))
    small = np.abs(z) < 1e-7
    zx = np.where(small, 1.0, z/np.where(small, 1.0, x))
    A = a/(fk*(1+(1-b)**2/24*lf**2+(1-b)**4/1920*lf**4))
    B = 1+((1-b)**2/24*a**2/(F*K)**(1-b)+rho*b*nu*a/(4*fk)+(2-3*rho**2)/24*nu**2)*T
    return A*zx*B

def sabr_mc(F0, K, T, r, a0, b, rho, nu, n, steps, scheme, cp, seed):
    """Monte Carlo for dF=a F^b dW1, da=nu a dW2, corr(dW1,dW2)=rho."""
    rng = np.random.default_rng(seed); dt = T/steps; sd = np.sqrt(dt)
    F = np.full(n, F0, float); a = np.full(n, a0, float)
    for _ in range(steps):
        z1 = rng.standard_normal(n); z2 = rho*z1+np.sqrt(1-rho**2)*rng.standard_normal(n)
        if scheme.startswith("Euler"):                       # full truncation
            Fp, ap = np.maximum(F, 0), np.maximum(a, 0)
            F = np.maximum(F+ap*Fp**b*sd*z1, 0); a = a+nu*ap*sd*z2
        else:                                                # log-transform
            Fp = np.maximum(F, 1e-12)
            F = Fp*np.exp(-.5*a**2*Fp**(2*b-2)*dt+a*Fp**(b-1)*sd*z1)
            a = a*np.exp(-.5*nu**2*dt+nu*sd*z2)
    pay = np.maximum(F-K, 0) if cp == "call" else np.maximum(K-F, 0)
    disc = np.exp(-r*T)*pay
    return F, disc.mean(), disc.std(ddof=1)/np.sqrt(n)

# ----------------------------------------------------------------- data / UI helpers
@st.cache_data(ttl=300, show_spinner=False)
def fetch_hist(t, n=6):
    try:
        import yfinance as yf
        h = yf.Ticker(t).history(period="1mo")["Close"].dropna()
        return [float(x) for x in h.iloc[-n:]] if len(h) else None
    except Exception:
        return None

def spot_widget(key, default=100.0):
    mode = st.radio("Spot source", ["Online", "Manual"], horizontal=True, key=f"spot_{key}_mode")
    if mode == "Online":
        t = st.text_input("Ticker", "AAPL", key=f"spot_{key}_ticker"); h = fetch_hist(t, 1)
        if h: st.success(f"{t}: {h[-1]:.2f}"); return h[-1], t
        st.warning("Could not fetch price online - enter it manually.")
        return st.number_input("Spot S", value=default, min_value=0.01, key=f"spot_{key}_fallback"), t
    return st.number_input("Spot S", value=default, min_value=0.01, key=f"spot_{key}_manual"), None

def help_btn(title, text, eqs):
    with st.popover("❓ Help"):
        st.markdown(f"### {title}\n{text}")
        for lab, e in eqs:
            st.markdown(f"**{lab}**"); st.latex(sp.latex(e))

def pdf_bytes(fig, tables):
    buf = io.BytesIO()
    with PdfPages(buf) as pdf:
        if fig is not None: pdf.savefig(fig)
        for title, df in tables:
            f, ax = plt.subplots(figsize=(11, 1+0.28*len(df))); ax.axis("off"); ax.set_title(title)
            tb = ax.table(cellText=df.round(4).astype(str).values, colLabels=list(df.columns), loc="center")
            tb.auto_set_font_size(False); tb.set_fontsize(7); pdf.savefig(f, bbox_inches="tight"); plt.close(f)
    return buf.getvalue()

_dl_n = [0]
def downloads(prefix, df, fig, tables):
    _dl_n[0] += 1; n = _dl_n[0]
    c1, c2 = st.columns(2)
    c1.download_button("⬇ CSV", df.to_csv(index=False), f"{prefix}.csv", "text/csv", key=f"{prefix}_csv{n}")
    c2.download_button("🖨 PDF (printable)", pdf_bytes(fig, tables), f"{prefix}.pdf", key=f"{prefix}_pdf{n}")

def set_demo(k): st.session_state[k] = int(np.random.randint(1, 2**31-1))
def close_demo(*ks):
    for k in ks: st.session_state[k] = None

def show_arb(lines):
    st.info(lines[0])
    if len(lines) > 1: (st.success if lines[1].startswith("No arbitrage") else st.warning)(lines[1])
    if len(lines) > 2:
        st.markdown("**Strategy**")
        for l in lines[2:]: st.markdown(l)

@st.cache_data(show_spinner=False)
def mc_cached(*args): return sabr_mc(*args)

# sympy symbols for the Help pages
S_, K_, r_, sg_, T_ = sp.symbols("S K r sigma T", positive=True)
al_, be_, rh_, nu_ = sp.symbols("alpha beta rho nu")
Nf, phif = sp.Function("N"), sp.Function("phi")
d1_ = (sp.log(S_/K_)+(r_+sg_**2/2)*T_)/(sg_*sp.sqrt(T_)); d2_ = d1_-sg_*sp.sqrt(T_)
BS_EQS = [("d1", sp.Eq(sp.Symbol("d_1"), d1_)), ("d2", sp.Eq(sp.Symbol("d_2"), d2_)),
          ("Call", sp.Eq(sp.Symbol("C"), S_*Nf(sp.Symbol("d_1"))-K_*sp.exp(-r_*T_)*Nf(sp.Symbol("d_2")))),
          ("Put", sp.Eq(sp.Symbol("P"), K_*sp.exp(-r_*T_)*Nf(-sp.Symbol("d_2"))-S_*Nf(-sp.Symbol("d_1"))))]

st.markdown("""<style>
.block-container{padding-top:2rem;max-width:1250px}
.hero{background:linear-gradient(90deg,#1f3c88,#3a7bd5);padding:1.2rem 1.6rem;border-radius:14px;margin-bottom:1rem}
.hero-title{color:#fff;font-size:1.9rem;font-weight:700}.hero-sub{color:#e6eefc;margin-top:.2rem}
div[data-testid="stMetric"]{background:rgba(58,123,213,.08);border:1px solid rgba(58,123,213,.25);padding:.8rem 1rem;border-radius:12px}
.stTabs [data-baseweb="tab-list"]{gap:.4rem}
.stTabs [data-baseweb="tab"]{background:rgba(58,123,213,.08);border-radius:10px 10px 0 0;padding:.5rem 1rem}
.stTabs [aria-selected="true"]{background:rgba(58,123,213,.25)}
.stButton>button,.stDownloadButton>button{border-radius:10px;font-weight:600}
</style>
<div class="hero"><div class="hero-title"> Options Toolkit</div>
<div class="hero-sub">Portfolio valuation · Arbitrage · Delta / Delta-Gamma hedging · SABR with Monte Carlo</div></div>""", unsafe_allow_html=True)
with st.sidebar:
    st.header("📈 Options Toolkit")
    st.caption("European options on non-dividend stocks. Each tab has a ❓ Help button (theory, assumptions, formulas) "
               "and a 🎲 button that runs a random demo.")
    st.button("✖ Close all random demos", on_click=close_demo, args=("demo1", "demo2", "demo3", "demo4"))
    st.markdown("**Tip:** use *Online* for live prices (yfinance) or *Manual* to type the spot yourself.")
tabs = st.tabs(["1 · Portfolio", "2 · Arbitrage", "3 · Delta / Delta-Gamma Hedge", "4 · SABR"])

# ================================================================= 1 PORTFOLIO
with tabs[0]:
    st.header("Portfolio Evaluation (European options)")
    st.button("🎲 Random portfolio demo", key="rnd1", on_click=set_demo, args=("demo1",)); out1 = st.container()
    help_btn("Portfolio valuation",
             "Each position is a European option on a non-dividend stock, valued with Black-Scholes (or a market price you "
             "supply). **Intrinsic value** is the payoff if exercised now. A long position has sign +1, a short position -1. "
             "Position value = sign × contracts × multiplier × option value. Assumes constant volatility and rate, lognormal "
             "prices, no dividends, no transaction costs.",
             [("Intrinsic value (call)", sp.Eq(sp.Symbol("I_c"), sp.Max(S_-K_, 0))),
              ("Intrinsic value (put)", sp.Eq(sp.Symbol("I_p"), sp.Max(K_-S_, 0))),
              ("Position value", sp.Eq(sp.Symbol("V_pos"), sp.Symbol("s")*sp.Symbol("n")*sp.Symbol("m")*sp.Symbol("V_BS")))]+BS_EQS)
    st.caption("Leave 'Spot' blank to fetch it online; fill it to override manually. Leave 'Market price' blank to use Black-Scholes.")
    df0 = pd.DataFrame([dict(Ticker="AAPL", Spot=None, Type="call", Position="long", Contracts=10, Multiplier=100,
                             K=200.0, T=0.5, Sigma=0.25, r=0.04, MarketPrice=None),
                        dict(Ticker="MSFT", Spot=None, Type="put", Position="short", Contracts=5, Multiplier=100,
                             K=400.0, T=0.25, Sigma=0.22, r=0.04, MarketPrice=None)])
    ed = st.data_editor(df0, num_rows="dynamic", use_container_width=True, key="pf", column_config={
        "Type": st.column_config.SelectboxColumn(options=["call", "put"]),
        "Position": st.column_config.SelectboxColumn(options=["long", "short"])})
    def show_portfolio(ed):
        rows = []
        for _, x in ed.dropna(subset=["Ticker"]).iterrows():
            S = x.Spot if pd.notna(x.Spot) else (fetch_hist(x.Ticker, 1) or [np.nan])[-1]
            if pd.isna(S): st.error(f"No spot for {x.Ticker}; enter it manually."); continue
            bsv = bs(S, x.K, x.r, x.Sigma, x["T"], x.Type)[0]
            px = x.MarketPrice if pd.notna(x.MarketPrice) else bsv
            sgn = 1 if x.Position == "long" else -1; units = sgn*x.Contracts*x.Multiplier
            iv = max(S-x.K, 0) if x.Type == "call" else max(x.K-S, 0)
            rows.append(dict(Ticker=x.Ticker, Spot=S, Type=x.Type, Position=x.Position, Units=units, K=x.K,
                             BS_price=bsv, Price_used=px, Intrinsic_per_opt=iv,
                             Intrinsic_value=units*iv, Market_value=units*px, Time_value=units*(px-iv)))
        if rows:
            res = pd.DataFrame(rows); st.dataframe(res.round(4), use_container_width=True)
            a, b = st.columns(2)
            a.metric("Total intrinsic value", f"{res.Intrinsic_value.sum():,.2f}")
            b.metric("Total market value", f"{res.Market_value.sum():,.2f}")
            fig, ax = plt.subplots(figsize=(8, 3.5)); w = .38; i = np.arange(len(res))
            ax.bar(i-w/2, res.Intrinsic_value, w, label="Intrinsic"); ax.bar(i+w/2, res.Market_value, w, label="Market")
            ax.set_xticks(i); ax.set_xticklabels(res.Ticker+" "+res.Type); ax.legend(); ax.set_title("Position values")
            st.pyplot(fig); downloads("portfolio", res, fig, [("Portfolio", res)])
    if st.button("Evaluate portfolio"): show_portfolio(ed)

if st.session_state.get("demo1"):
    rg = np.random.default_rng(st.session_state["demo1"]); rows_ = []
    for tk in rg.choice(["AAPL", "MSFT", "TSLA", "NVDA", "AMZN", "GOOG", "META"], 4, replace=False):
        S0 = round(float(rg.uniform(50, 500)), 2)
        rows_.append(dict(Ticker=str(tk), Spot=S0, Type=str(rg.choice(["call", "put"])), Position=str(rg.choice(["long", "short"])),
                          Contracts=int(rg.integers(1, 20)), Multiplier=100, K=round(S0*float(rg.uniform(.85, 1.15)), 2),
                          T=round(float(rg.uniform(.1, 1.5)), 2), Sigma=round(float(rg.uniform(.15, .6)), 2),
                          r=round(float(rg.uniform(.01, .05)), 3), MarketPrice=None))
    with out1:
        st.button("✖ Close demo", key="close1", on_click=close_demo, args=("demo1",))
        st.caption("Random portfolio (random spots, no online fetch):"); rd = pd.DataFrame(rows_)
        st.dataframe(rd, use_container_width=True); show_portfolio(rd)

# ================================================================= 2 ARBITRAGE
with tabs[1]:
    st.header("Arbitrage Calculator")
    st.button("🎲 Random arbitrage case", key="rnd2", on_click=set_demo, args=("demo2",)); out2 = st.container()
    help_btn("Synthetic replication & parity",
             "An option can be replicated by Δ shares plus a bond position (Black-Scholes replicating portfolio). If the market "
             "price differs from the replication cost, buy the cheap side and sell the expensive side: the payoffs cancel "
             "and the price gap is risk-free profit. Put-call parity gives a model-free static check for European options "
             "on non-dividend stocks. Assumes frictionless markets, constant σ and r, continuous rebalancing.",
             [("Replicating portfolio (call)", sp.Eq(sp.Symbol("C"), Nf(sp.Symbol("d_1"))*S_+(-K_*sp.exp(-r_*T_)*Nf(sp.Symbol("d_2"))))),
              ("Replicating portfolio (put)", sp.Eq(sp.Symbol("P"), -Nf(-sp.Symbol("d_1"))*S_+K_*sp.exp(-r_*T_)*Nf(-sp.Symbol("d_2")))),
              ("Put-call parity", sp.Eq(sp.Symbol("C")-sp.Symbol("P"), S_-K_*sp.exp(-r_*T_)))])
    c1, c2 = st.columns(2)
    with c1: S2, _ = spot_widget("a")
    with c2:
        K2a = st.number_input("Strike K", value=round(S2), key=f"ak_{round(S2)}"); T2a = st.number_input("T (years)", value=0.5, min_value=0.01, key="at")
        r2a = st.number_input("Risk-free r", value=0.04, key="ar"); s2a = st.number_input("Volatility σ", value=0.25, min_value=0.01, key="as")
    cp2 = st.radio("Option type", ["call", "put"], horizontal=True, key="acp")
    fair = bs(S2, K2a, r2a, s2a, T2a, cp2)[0]
    mk = st.number_input(f"Observed market price (model value {fair:.4f})", value=float(round(fair*1.1, 4)), key=f"amk_{round(fair, 4)}")
    if st.button("Find arbitrage (synthetic option)"):
        show_arb(arb_steps(S2, K2a, r2a, s2a, T2a, cp2, mk)[0])
    st.subheader("Put-call parity check")
    pa, pb = st.columns(2)
    Cm = pa.number_input("Market call price", value=10.0, key="pc"); Pm = pb.number_input("Market put price", value=6.0, key="pp")
    def parity(S2, K2a, r2a, T2a, Cm, Pm):
        pv = K2a*np.exp(-r2a*T2a); gap = (Cm-Pm)-(S2-pv)
        if abs(gap) < 1e-4: st.success("Parity holds - no arbitrage.")
        elif gap > 0:
            st.warning(f"Call is rich by **{gap:.4f}**. 1. Sell call, 2. Buy put, 3. Buy stock at {S2:.2f}, "
                        f"4. Borrow PV(K) = {pv:.4f}. Upfront cash +{gap:.4f}; at T the position nets to 0 whatever the final stock price is.")
        else:
            st.warning(f"Call is cheap by **{-gap:.4f}**. 1. Buy call, 2. Sell put, 3. Short stock at {S2:.2f}, "
                        f"4. Lend PV(K) = {pv:.4f}. Upfront cash +{-gap:.4f}; at T the position nets to 0 whatever the final stock price is.")

    if st.button("Check parity"): parity(S2, K2a, r2a, T2a, Cm, Pm)

if st.session_state.get("demo2"):
    rg = np.random.default_rng(st.session_state["demo2"]); S_r = round(float(rg.uniform(50, 400)), 2); K_r = round(S_r*float(rg.uniform(.9, 1.1)), 2)
    r_r = round(float(rg.uniform(.01, .06)), 3); sg_r = round(float(rg.uniform(.15, .5)), 2); T_r = round(float(rg.uniform(.2, 1.5)), 2)
    cp_r = str(rg.choice(["call", "put"])); fair_r = bs(S_r, K_r, r_r, sg_r, T_r, cp_r)[0]
    mk_r = round(fair_r*(1+float(rg.choice([-1, 1])*rg.uniform(.08, .3))), 4)
    with out2:
        st.button("✖ Close demo", key="close2", on_click=close_demo, args=("demo2",))
        st.caption(f"Random case: S={S_r}, K={K_r}, r={r_r}, sigma={sg_r}, T={T_r}, {cp_r}")
        show_arb(arb_steps(S_r, K_r, r_r, sg_r, T_r, cp_r, mk_r)[0])
        st.markdown("**Put-call parity with random quotes:**")
        Cr = bs(S_r, K_r, r_r, sg_r, T_r, "call")[0]; Pr = bs(S_r, K_r, r_r, sg_r, T_r, "put")[0]+float(rg.uniform(-1.5, 1.5))
        st.caption(f"Call = {Cr:.4f}, Put = {Pr:.4f}"); parity(S_r, K_r, r_r, T_r, round(Cr, 4), round(Pr, 4))

# ================================================================= 3 HEDGING
def run_hedge(path, K, r, sig, T, cp, q, K2, T2, use_gamma):
    dt = 1/252; sh = u = cash = 0.0; rows = []; V0 = None
    for i, S in enumerate(path):
        V, d, g = bs(S, K, r, sig, T-i*dt, cp); Vh, dh, gh = bs(S, K2, r, sig, T2-i*dt, cp)
        V0 = V if V0 is None else V0
        cash *= np.exp(r*dt)
        pnl = q*V+sh*S+u*Vh+cash-q*V0                         # before rebalancing
        un = -q*g/gh if (use_gamma and gh > 1e-12) else 0.0
        shn = -(q*d+un*dh)
        cash -= (shn-sh)*S+(un-u)*Vh
        rows.append({"Day": i, "Kind": "Start" if i == 0 else "-", "Spot": S, "T_left": T-i*dt, "Option": V, "Delta": d,
                     "Gamma": g, "Shares_held": shn, "HedgeOpt_held": un, "Trade_shares": shn-sh, "Trade_hedgeOpt": un-u,
                     "Cash": cash, "PnL_hedged": pnl, "PnL_unhedged": q*(V-V0)})
        sh, u = shn, un
    return pd.DataFrame(rows)

with tabs[2]:
    st.header("Delta & Delta-Gamma Hedging")
    st.button("🎲 Random hedging scenario", key="rnd3", on_click=set_demo, args=("demo3",)); out3 = st.container()
    help_btn("Hedging",
             "**Delta hedge:** hold −Δ shares per option so the portfolio is insensitive to small spot moves. "
             "**Delta-gamma hedge:** also trade a second option (strike K₂, maturity T₂) to cancel gamma, then fix delta with "
             "shares. Positions are re-set every day (252 trading days/yr); cash accrues at r, so the strategy is self-financing "
             "and P&L = portfolio value − initial value. Assumes Black-Scholes prices, constant σ and r, no transaction costs.",
             [("Delta", sp.Eq(sp.Symbol("Delta"), Nf(sp.Symbol("d_1")))),
              ("Gamma", sp.Eq(sp.Symbol("Gamma"), phif(sp.Symbol("d_1"))/(S_*sg_*sp.sqrt(T_)))),
              ("Hedge-option units", sp.Eq(sp.Symbol("n_h"), -sp.Symbol("q")*sp.Symbol("Gamma")/sp.Symbol("Gamma_h"))),
              ("Shares", sp.Eq(sp.Symbol("n_S"), -(sp.Symbol("q")*sp.Symbol("Delta")+sp.Symbol("n_h")*sp.Symbol("Delta_h")))),
              ("Daily P&L", sp.Eq(sp.Symbol("PnL_t"), sp.Symbol("V^pf_t")-sp.Symbol("V^pf_0")))]+BS_EQS)
    c1, c2, c3 = st.columns(3)
    with c1: S3, tick3 = spot_widget("h")
    with c2:
        K3 = st.number_input("Strike K", value=round(S3), key=f"hk_{round(S3)}"); r3 = st.number_input("r", value=0.04, key="hr")
        sg3 = st.number_input("σ", value=0.25, min_value=0.01, key="hs"); T3 = st.number_input("T (years)", value=0.5, min_value=0.05, key="ht")
    with c3:
        cp3 = st.radio("Type", ["call", "put"], horizontal=True, key="hcp"); pos3 = st.radio("Position", ["short", "long"], horizontal=True, key="hpos")
        nopt = st.number_input("Number of options", value=1000, min_value=1, key="hn")
        K3b = st.number_input("Gamma-hedge option strike K₂", value=round(S3*1.05), key=f"hk2_{round(S3)}"); T3b = st.number_input("Its maturity T₂", value=T3+0.25, key=f"ht2_{T3}")
    q = nopt*(1 if pos3 == "long" else -1)
    nd = st.slider("Days to simulate / report", 1, 5, 5)
    src = st.radio("Spot path source", ["Manual", "Random (GBM forecast)", "Historical (online backtest)"], horizontal=True)
    if src == "Manual":
        cols = st.columns(nd); path = [S3]+[cols[i].number_input(f"Day {i+1}", value=float(S3), key=f"hm{i}_{round(S3, 2)}") for i in range(nd)]; kind = "Forecast"
    elif src.startswith("Random"):
        a, b, c = st.columns(3); mu = a.number_input("Drift μ", value=0.05); vs = b.number_input("Sim vol", value=sg3); seed = int(c.number_input("Seed", value=1))
        z = np.random.default_rng(seed).standard_normal(nd); path = [S3]
        for zi in z: path.append(path[-1]*np.exp((mu-.5*vs**2)/252+vs/np.sqrt(252)*zi))
        kind = "Forecast"
    else:
        path = fetch_hist(tick3, nd+1) if tick3 else None; kind = "Historical"
        if not path or len(path) < nd+1: st.error("Historical data unavailable (need Online mode with a valid ticker)."); path = None
    def show_hedge(path, K3, r3, sg3, T3, cp3, q, K3b, T3b, kind):
        d_only = run_hedge(path, K3, r3, sg3, T3, cp3, q, K3b, T3b, False); dg = run_hedge(path, K3, r3, sg3, T3, cp3, q, K3b, T3b, True)
        for d in (d_only, dg): d["Kind"] = [("Start" if i == 0 else kind) for i in range(len(d))]
        st.subheader("Delta hedge - daily adjustments"); st.dataframe(d_only.round(4), use_container_width=True)
        st.subheader("Delta-Gamma hedge - daily adjustments"); st.dataframe(dg.round(4), use_container_width=True)
        summ = pd.DataFrame({"Strategy": ["Unhedged", "Delta", "Delta-Gamma"],
                             "Final P&L": [d_only.PnL_unhedged.iloc[-1], d_only.PnL_hedged.iloc[-1], dg.PnL_hedged.iloc[-1]]})
        st.table(summ.round(2))
        fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
        ax[0].plot(d_only.Day, d_only.Spot, "o-"); ax[0].set_title(f"Spot path ({kind})"); ax[0].set_xlabel("Day")
        for lab, col, d in [("Unhedged", "PnL_unhedged", d_only), ("Delta", "PnL_hedged", d_only), ("Delta-Gamma", "PnL_hedged", dg)]:
            ax[1].plot(d.Day, d[col], "o-", label=lab)
        ax[1].set_title("P&L"); ax[1].set_xlabel("Day"); ax[1].legend(); st.pyplot(fig)
        full = pd.concat([d_only.assign(Strategy="Delta"), dg.assign(Strategy="Delta-Gamma")])
        downloads("hedging", full, fig, [("Summary", summ), ("Delta hedge", d_only), ("Delta-Gamma hedge", dg)])

    if path and st.button("Run hedging simulation"): show_hedge(path, K3, r3, sg3, T3, cp3, q, K3b, T3b, kind)

if st.session_state.get("demo3"):
    rg = np.random.default_rng(st.session_state["demo3"]); S_h = round(float(rg.uniform(50, 400)), 2); sg_h = round(float(rg.uniform(.2, .5)), 2)
    T_h = round(float(rg.uniform(.3, 1.0)), 2); cp_h = str(rg.choice(["call", "put"])); q_h = -100*int(rg.integers(5, 20))
    path_h = [S_h]
    for z in rg.standard_normal(5): path_h.append(path_h[-1]*np.exp(-.5*(2*sg_h)**2/252+2*sg_h/np.sqrt(252)*z))
    with out3:
        st.button("✖ Close demo", key="close3", on_click=close_demo, args=("demo3",))
        st.caption(f"Random scenario: short {-q_h} {cp_h}s, S={S_h}, K={round(S_h)}, sigma={sg_h}, T={T_h}; "
                   f"random 5-day GBM path with 2x volatility so the moves are visible.")
        show_hedge(path_h, round(S_h), 0.04, sg_h, T_h, cp_h, q_h, round(S_h*1.05), T_h+.25, "Forecast")

# ================================================================= 4 SABR
def load_calib(f):
    df = pd.read_csv(f) if f.name.lower().endswith(".csv") else pd.read_excel(f)
    df.columns = [c.strip().lower() for c in df.columns]
    kc = next(c for c in df.columns if "strike" in c or c == "k")
    vc = next((c for c in df.columns if c in ("vol", "iv", "impliedvol", "implied_vol", "implied vol", "volatility")), None)
    return df, kc, vc

with tabs[3]:
    st.header("SABR Calculator (Hagan) + Monte Carlo")
    help_btn("SABR model",
             "SABR models the forward F and its stochastic volatility α with correlated shocks. Hagan's expansion gives an "
             "implied Black volatility σ_B(K) in closed form, producing a smile. For equities set **β = 1** (lognormal). "
             "Forward for a non-dividend stock: F = S·e^{rT}. **Calibration:** least-squares fit of (α, ρ, ν) to market "
             "implied vols with β fixed. **Monte Carlo:** simulate the SDEs on a time grid with M paths; Euler-Maruyama with "
             "*full truncation* (negatives replaced by 0 inside the coefficients) or the *log-transform* scheme (exact "
             "lognormal step for α, log-Euler for F, which keeps F>0). The option price is the discounted average payoff "
             "of F_T = S_T. The expansion is an approximation (weak for very low strikes / long maturities); ρ∈(−1,1).",
             [("Forward", sp.Eq(sp.Symbol("F"), S_*sp.exp(r_*T_))),
              ("SDE for F", sp.Eq(sp.Symbol("dF_t"), sp.Symbol("alpha_t")*sp.Symbol("F_t")**be_*sp.Symbol("dW^1_t"))),
              ("SDE for alpha", sp.Eq(sp.Symbol("d alpha_t"), nu_*sp.Symbol("alpha_t")*sp.Symbol("dW^2_t"))),
              ("Correlation", sp.Eq(sp.Symbol("d<W^1,W^2>_t"), rh_*sp.Symbol("dt"))),
              ("Hagan implied vol", sp.Eq(sp.Symbol("sigma_B"), al_/((S_*K_)**((1-be_)/2)*(1+(1-be_)**2/24*sp.log(S_/K_)**2+(1-be_)**4/1920*sp.log(S_/K_)**4))
                                          *sp.Symbol("z")/sp.Symbol("x(z)")*(1+((1-be_)**2/24*al_**2/(S_*K_)**(1-be_)+rh_*be_*nu_*al_/(4*(S_*K_)**((1-be_)/2))+(2-3*rh_**2)/24*nu_**2)*T_))),
              ("z", sp.Eq(sp.Symbol("z"), nu_/al_*(S_*K_)**((1-be_)/2)*sp.log(S_/K_))),
              ("x(z)", sp.Eq(sp.Symbol("x(z)"), sp.log((sp.sqrt(1-2*rh_*sp.Symbol("z")+sp.Symbol("z")**2)+sp.Symbol("z")-rh_)/(1-rh_)))),
              ("Euler-Maruyama (full truncation)", sp.Eq(sp.Symbol("F_{t+dt}"), sp.Symbol("F_t^+")+sp.Symbol("alpha_t^+")*sp.Symbol("(F_t^+)")**be_*sp.sqrt(sp.Symbol("dt"))*sp.Symbol("Z_1"))),
              ("Log-transform", sp.Eq(sp.Symbol("F_{t+dt}"), sp.Symbol("F_t")*sp.exp(-sp.Symbol("alpha_t")**2*sp.Symbol("F_t")**(2*be_-2)*sp.Symbol("dt")/2+sp.Symbol("alpha_t")*sp.Symbol("F_t")**(be_-1)*sp.sqrt(sp.Symbol("dt"))*sp.Symbol("Z_1")))),
              ("MC price", sp.Eq(sp.Symbol("V_0"), sp.exp(-r_*T_)*sp.Symbol("1/M sum_i")*sp.Max(sp.Symbol("F_T^i")-K_, 0)))])
    st.button("🎲 Random SABR demo (random market, calibrate, Monte Carlo)", key="rnd4", on_click=set_demo, args=("demo4",)); out4 = st.container()
    c1, c2 = st.columns(2)
    with c1: S4, _ = spot_widget("s")
    with c2:
        r4 = st.number_input("r", value=0.04, key="sr_"); T4 = st.number_input("T (years)", value=0.5, min_value=0.02, key="st_")
        cp4 = st.radio("Option type", ["call", "put"], horizontal=True, key="scp")
    F4 = S4*np.exp(r4*T4); st.info(f"Forward F = S·e^(rT) = {F4:.4f}")
    st.subheader("Calibration (β fixed; equities β = 1)")
    beta = st.number_input("β", 0.0, 1.0, 1.0, key="sb")
    _rg = np.random.default_rng(7); _Ks = np.linspace(.8*S4, 1.2*S4, 13)
    _v = hagan(F4, _Ks, T4, .30, 1.0, -.4, .6)*(1+_rg.normal(0, .005, len(_Ks)))
    st.download_button("⬇ Sample calibration CSV (matches current S, r, T)", pd.DataFrame({"Strike": _Ks.round(2), "Vol": _v.round(5)}).to_csv(index=False),
                       "sabr_sample.csv", "text/csv", key="sabr_sample")
    up = st.file_uploader("CSV/Excel with columns: Strike, Vol (implied vol, decimal)  [optional: Price]", type=["csv", "xlsx", "xls"])
    if up and st.button("Calibrate α, ρ, ν"):
        try:
            df, kc, vc = load_calib(up); Ks = df[kc].values.astype(float)
            if vc: vols = df[vc].values.astype(float)
            else:
                pc = next(c for c in df.columns if "price" in c)
                vols = np.array([brentq(lambda s: bs(S4, k, r4, s, T4, cp4)[0]-p, 1e-4, 5) for k, p in zip(Ks, df[pc])])
            atm = vols[np.argmin(abs(Ks-F4))]
            f = lambda p: hagan(F4, Ks, T4, p[0], beta, p[1], p[2])-vols
            sol = least_squares(f, [atm*F4**(1-beta), -.3, .5], bounds=([1e-4, -.999, 1e-4], [10, .999, 10]))
            st.session_state.update(sa=float(sol.x[0]), sr=float(sol.x[1]), sn=float(sol.x[2]))
            st.success(f"Calibrated: α={sol.x[0]:.4f}, ρ={sol.x[1]:.4f}, ν={sol.x[2]:.4f}  (RMSE {np.sqrt(np.mean(sol.fun**2)):.5f})")
        except Exception as e:
            st.error(f"Could not calibrate: {e}")
    p1, p2, p3 = st.columns(3)
    for _k, _v in (("sa", 0.25), ("sr", -0.3), ("sn", 0.5)): st.session_state.setdefault(_k, _v)
    alpha = p1.number_input("α", min_value=0.0001, format="%.4f", key="sa")
    rho = p2.number_input("ρ", min_value=-0.999, max_value=0.999, format="%.4f", key="sr")
    nu = p3.number_input("ν", min_value=0.0001, format="%.4f", key="sn")

    st.subheader("Volatility smile across (S, F) combinations")
    st.caption("Each spot S gives its own forward F = S·e^(rT). Enter several spots separated by commas.")
    spots = [float(x) for x in st.text_input("Spots", f"{S4*.9:.2f}, {S4:.2f}, {S4*1.1:.2f}").split(",") if x.strip()]
    Kgrid = np.linspace(.7*S4, 1.3*S4, 61); rows = []
    fig, ax = plt.subplots(figsize=(8, 4))
    for s in spots:
        Fs = s*np.exp(r4*T4); ax.plot(Kgrid, hagan(Fs, Kgrid, T4, alpha, beta, rho, nu), label=f"S={s:.1f}, F={Fs:.1f}")
        atm = float(hagan(Fs, Fs, T4, alpha, beta, rho, nu))
        for k in np.linspace(.8*s, 1.2*s, 9):
            v = float(hagan(Fs, k, T4, alpha, beta, rho, nu))
            rows.append(dict(S=s, F=Fs, K=k, Hagan_vol=v, SABR_price=bs(s, k, r4, v, T4, cp4)[0],
                             BS_flat_ATM_vol=atm, BS_price=bs(s, k, r4, atm, T4, cp4)[0]))
    ax.set_xlabel("Strike"); ax.set_ylabel("Implied vol"); ax.set_title("SABR volatility smile"); ax.legend(); st.pyplot(fig)
    tab = pd.DataFrame(rows); tab["Diff"] = tab.SABR_price-tab.BS_price
    par = pd.DataFrame({"Parameter": ["alpha", "beta", "rho", "nu", "T", "r"], "Value": [alpha, beta, rho, nu, T4, r4]})
    st.dataframe(tab.round(4), use_container_width=True); st.table(par)
    downloads("sabr", tab, fig, [("Calibrated parameters", par), ("Prices / forwards / BS", tab)])

    st.subheader("Mispriced option -> arbitrage with SABR price")
    m1, m2 = st.columns(2)
    Km = m1.number_input("Strike", value=float(round(S4)), key=f"mk_k_{round(S4)}")
    v = float(hagan(F4, Km, T4, alpha, beta, rho, nu)); model = bs(S4, Km, r4, v, T4, cp4)[0]
    Pm_ = m2.number_input(f"Market price (SABR model price {model:.4f}, Hagan vol {v:.2%})", value=float(round(model*1.1, 4)), key=f"mk_p_{round(model, 4)}")
    if st.button("Find SABR arbitrage"):
        show_arb(arb_steps(S4, Km, r4, v, T4, cp4, Pm_)[0])

    st.subheader("Monte Carlo simulation of the SABR SDEs")
    g1, g2, g3, g4 = st.columns(4)
    nsim = int(g1.number_input("Number of simulations", 100, 5_000_000, 100_000, step=10_000)); nst = int(g2.number_input("Time steps", 5, 2000, 100))
    scheme = g3.selectbox("Scheme", ["Log-transform (full-truncation safe)", "Euler-Maruyama (full truncation)"]); sd_ = int(g4.number_input("Seed", value=42))
    Kmc = st.number_input("Strike for MC", value=float(round(S4)), key=f"mck_{round(S4)}")
    if st.button("Run Monte Carlo"):
        with st.spinner("Simulating..."):
            FT, price, se = sabr_mc(F4, Kmc, T4, r4, alpha, beta, rho, nu, nsim, nst, scheme, cp4, sd_)
        hv = float(hagan(F4, Kmc, T4, alpha, beta, rho, nu)); hp = bs(S4, Kmc, r4, hv, T4, cp4)[0]
        a, b, c = st.columns(3)
        a.metric("Mean final price S_T", f"{FT.mean():.4f}", f"forward {F4:.4f}"); b.metric("MC option price (PV of mean payoff)", f"{price:.4f}", f"± {1.96*se:.4f} (95%)")
        c.metric("Hagan/BS price", f"{hp:.4f}")
        fg, ax = plt.subplots(figsize=(7, 3.5)); ax.hist(FT, bins=80, alpha=.8); ax.axvline(Kmc, color="r", ls="--", label="Strike")
        ax.set_title("Distribution of simulated S_T"); ax.legend(); st.pyplot(fg)
        mc = pd.DataFrame([dict(Sims=nsim, Steps=nst, Scheme=scheme, Mean_ST=FT.mean(), MC_price=price, Std_err=se, Hagan_price=hp)])
        downloads("sabr_mc", mc, fg, [("Monte Carlo", mc)])

if st.session_state.get("demo4"):
    rg = np.random.default_rng(st.session_state["demo4"]); S_s = round(float(rg.uniform(50, 400)), 2); r_s = round(float(rg.uniform(.01, .05)), 3)
    T_s = round(float(rg.uniform(.25, 1.5)), 2); F_s = S_s*np.exp(r_s*T_s)
    true = (float(rg.uniform(.15, .5)), float(rg.uniform(-.7, -.1)), float(rg.uniform(.3, 1.0)))
    Ks = np.linspace(.8*F_s, 1.2*F_s, 15); mv = hagan(F_s, Ks, T_s, true[0], 1.0, true[1], true[2])*(1+rg.normal(0, .004, len(Ks)))
    sol = least_squares(lambda p: hagan(F_s, Ks, T_s, p[0], 1.0, p[1], p[2])-mv, [mv[7], -.3, .5], bounds=([1e-4, -.999, 1e-4], [10, .999, 10]))
    a_c, r_c, n_c = sol.x; Kc = round(F_s, 2)
    FT, px, se = mc_cached(F_s, Kc, T_s, r_s, a_c, 1.0, r_c, n_c, 100000, 100, "Log-transform", "call", 1)
    hp = bs(S_s, Kc, r_s, float(hagan(F_s, Kc, T_s, a_c, 1.0, r_c, n_c)), T_s, "call")[0]
    with out4:
        st.button("✖ Close demo", key="close4", on_click=close_demo, args=("demo4",))
        st.caption(f"Random market: S={S_s}, r={r_s}, T={T_s}, F={F_s:.2f}, beta=1 (noisy implied vols generated from hidden SABR parameters)")
        st.table(pd.DataFrame({"Parameter": ["alpha", "rho", "nu"], "True (hidden)": list(true), "Calibrated": [a_c, r_c, n_c]}).round(4))
        fg, ax = plt.subplots(figsize=(7, 3.5)); kk = np.linspace(Ks[0], Ks[-1], 80)
        ax.plot(Ks, mv, "o", label="Random market vols"); ax.plot(kk, hagan(F_s, kk, T_s, a_c, 1.0, r_c, n_c), label="Calibrated SABR")
        ax.set_xlabel("Strike"); ax.set_ylabel("Implied vol"); ax.legend(); st.pyplot(fg)
        m1, m2, m3 = st.columns(3)
        m1.metric(f"Hagan/BS call price (K={Kc})", f"{hp:.4f}"); m2.metric("MC call price (100k sims)", f"{px:.4f}", f"± {1.96*se:.4f}")
        m3.metric("Mean final price S_T", f"{FT.mean():.4f}", f"forward {F_s:.4f}")
