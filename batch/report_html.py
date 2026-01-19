# /home/ssm-user/jupyter/batch/report_html.py
from __future__ import annotations

from typing import Dict, List, Any
import pandas as pd


def df_to_html_table(df: pd.DataFrame, max_rows: int = 200) -> str:
    dfx = df.copy()
    if len(dfx) > max_rows:
        dfx = dfx.head(max_rows)
    return dfx.to_html(index=False, escape=False, border=0)


def inject_floating_toc(html_text: str) -> str:
    """
    HTML에 우측 상단 고정 TOC 삽입 (h1/h2/h3 기반)
    - header에 id 없으면 자동 부여
    - 스타일도 head에 주입
    """
    try:
        from bs4 import BeautifulSoup
    except Exception:
        # bs4 없으면 그냥 원본 리턴
        return html_text

    soup = BeautifulSoup(html_text, "html.parser")

    if not soup.head:
        head = soup.new_tag("head")
        if soup.html:
            soup.html.insert(0, head)
        else:
            soup.insert(0, head)

    if not soup.body:
        body = soup.new_tag("body")
        if soup.html:
            soup.html.append(body)
        else:
            soup.append(body)

    toc_container = soup.new_tag("div", id="toc")
    toc_title_tag = soup.new_tag("strong")
    toc_title_tag.string = "목차"
    toc_container.append(toc_title_tag)

    toc_list = soup.new_tag("ul")
    toc_container.append(toc_list)

    header_tags = soup.find_all(["h1", "h2", "h3"])
    current_h1 = None
    current_h2 = None

    for idx, header in enumerate(header_tags):
        if not header.has_attr("id"):
            header["id"] = f"toc_{idx}"

        link = soup.new_tag("a", href=f"#{header['id']}")
        link.string = header.get_text(strip=True)

        list_item = soup.new_tag("li")
        list_item.append(link)

        if header.name == "h1":
            toc_list.append(list_item)
            current_h1 = list_item
            current_h2 = None

        elif header.name == "h2":
            if current_h1 is None:
                toc_list.append(list_item)
            else:
                if not current_h1.find("ul"):
                    current_h1.append(soup.new_tag("ul"))
                current_h1.find("ul").append(list_item)
            current_h2 = list_item

        elif header.name == "h3":
            if current_h2 is not None:
                if not current_h2.find("ul"):
                    current_h2.append(soup.new_tag("ul"))
                current_h2.find("ul").append(list_item)
            elif current_h1 is not None:
                if not current_h1.find("ul"):
                    current_h1.append(soup.new_tag("ul"))
                current_h1.find("ul").append(list_item)
            else:
                toc_list.append(list_item)

    style_tag = soup.new_tag("style")
    style_tag.string = """
#toc {
    position: fixed;
    top: 20px;
    right: 20px;
    width: 250px;
    background: #f9f9f9;
    border: 1px solid #ddd;
    padding: 10px;
    max-height: 90vh;
    overflow-y: auto;
    box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    z-index: 1000;
    font-family: sans-serif;
    font-size: 14px;
}
#toc ul {
    list-style: none;
    padding-left: 0;
    margin: 8px 0 0 0;
}
#toc li {
    margin: 5px 0;
}
#toc li ul {
    padding-left: 15px;
}
#toc li ul li ul {
    padding-left: 15px;
}
#toc a {
    text-decoration: none;
    color: #333;
}
#toc a:hover {
    text-decoration: underline;
}

/* TOC 가리는 만큼 본문 오른쪽 여백 확보(겹침 방지) */
body {
    margin-right: 300px;
}
"""
    soup.head.append(style_tag)

    soup.body.insert(0, toc_container)

    return str(soup)


def build_report_context(
    *,
    building_id: int,
    building_name: str,
    start_date: str,
    end_date: str,
    run_id: str,
    now_kst: str,
    date_range_df: pd.DataFrame,
    figs: Dict[str, str],
    top5_df: pd.DataFrame,
    kw_imgs: List[str],
    wc_img: str,
    hm_imgs: List[str],
    is_empty: bool = False,
) -> Dict[str, Any]:
    report_month = start_date.replace("-", "")[:6]
    report_month_fmt = f"{report_month[:4]}.{report_month[4:]}" if len(report_month) == 6 else report_month

    return {
        "building_id": building_id,
        "building_name": building_name,
        "start_date": start_date,
        "end_date": end_date,
        "run_id": run_id,
        "now_kst": now_kst,
        "report_month_fmt": report_month_fmt,
        "date_range_df": date_range_df,
        "figs": figs,
        "top5_df": top5_df,
        "kw_imgs": kw_imgs,
        "wc_img": wc_img,
        "hm_imgs": hm_imgs,
        "is_empty": is_empty,
    }


def render_report_html(ctx: Dict[str, Any]) -> str:
    html: List[str] = []
    html.append("<!DOCTYPE html>")
    html.append("<html lang='ko'><head><meta charset='utf-8'/>")
    html.append("<meta name='viewport' content='width=device-width, initial-scale=1'/>")
    html.append("<title>월간 VOC 분석 보고서</title>")
    html.append(
        """
        <style>
          body { font-family: Arial, Helvetica, sans-serif; margin: 24px; }
          h1 { margin-top: 28px; }
          h2 { margin-top: 22px; }
          h3 { margin-top: 16px; }
          .meta { color: #444; margin-bottom: 10px; }
          .img { margin: 10px 0 18px 0; }
          img { max-width: 100%; height: auto; border: 1px solid #ddd; }
          table { border-collapse: collapse; width: 100%; margin: 10px 0 18px 0; }
          th, td { border: 1px solid #ddd; padding: 8px; font-size: 13px; vertical-align: top; }
          th { background: #f5f5f5; }

          .report-title-wrap {
              position: relative;
              left: 50%;
              transform: translateX(-50%);
              width: 100vw;
              display: flex;
              justify-content: center;
              align-items: center;
              padding: 18px 0 8px 0;
              margin: 0 0 10px 0;
            }
          .report-title {
            font-size: 44px;
            font-weight: 900;
            letter-spacing: -0.8px;
            line-height: 1.15;
            text-align: center;
          }
        </style>
        """
    )
    html.append("</head><body>")

    html.append("""
      <div class="report-title-wrap">
        <div class="report-title">월간 VOC 분석 보고서</div>
      </div>
    """)

    # 빈 데이터 경고 배너
    if ctx.get("is_empty", False):
        html.append(f"""
        <div style="background-color: #ffcccc; border: 2px solid #cc0000; color: #cc0000;
                    padding: 15px; margin: 20px 0; border-radius: 5px; text-align: center;
                    font-size: 16px; font-weight: bold;">
            해당 기간({ctx['start_date']} ~ {ctx['end_date']}) VOC 데이터가 없습니다.
        </div>
        """)

    html.append("<h1>분석 보고서 개요</h1>")
    html.append(f"<div class='meta'><b>현장명</b> : {ctx['building_name']}</div>")
    html.append(f"<div class='meta'><b>보고대상월</b> : {ctx['report_month_fmt']}</div>")
    html.append(f"<div class='meta'>생성시각(KST): {ctx['now_kst']}</div>")

    html.append("<div class='meta'>")
    html.append(
        f"BUILDING_ID={ctx['building_id']} / START_DATE={ctx['start_date']} / "
        f"END_DATE={ctx['end_date']} / RUN_ID={ctx['run_id']}"
    )
    html.append("</div>")

    html.append("<h1 id='basic'>대상 데이터 기초 분석</h1>")

    html.append("<h2>날짜 범위</h2>")
    date_range_df = ctx["date_range_df"].reset_index().rename(columns={"index": "구분"})
    html.append(df_to_html_table(date_range_df))

    html.append("<h2>일별 VOC 발생 건수</h2>")
    html.append(f"<div class='img'><img src='data:image/png;base64,{ctx['figs']['daily_counts']}'/></div>")

    html.append("<h2>층별 VOC 발생 건수</h2>")
    html.append(f"<div class='img'><img src='data:image/png;base64,{ctx['figs']['floor_counts']}'/></div>")

    html.append("<h2>팀별 VOC 처리 건수</h2>")
    html.append(f"<div class='img'><img src='data:image/png;base64,{ctx['figs']['team_counts']}'/></div>")

    html.append("<h2>요일별 VOC 발생 건수</h2>")
    html.append(f"<div class='img'><img src='data:image/png;base64,{ctx['figs']['weekday_counts']}'/></div>")

    html.append("<h2>처리 소요 시간</h2>")
    html.append("<h3>처리소요시간 TOP 5</h3>")
    html.append(df_to_html_table(ctx["top5_df"]))

    html.append("<h3>일별 평균 처리 소요 시간</h3>")
    html.append(f"<div class='img'><img src='data:image/png;base64,{ctx['figs']['avg_processing']}'/></div>")

    html.append("<h1 id='keyword'>키워드 분석</h1>")
    html.append("<h2>키워드 빈도 상위 20</h2>")
    for b64 in ctx["kw_imgs"]:
        html.append(f"<div class='img'><img src='data:image/png;base64,{b64}'/></div>")

    html.append("<h1 id='wordcloud'>워드 클라우드</h1>")
    html.append(f"<div class='img'><img src='data:image/png;base64,{ctx['wc_img']}'/></div>")

    html.append("<h1 id='topic'>주제분석</h1>")
    for b64 in ctx["hm_imgs"]:
        html.append(f"<div class='img'><img src='data:image/png;base64,{b64}'/></div>")

    html.append("</body></html>")

    final_html = "\n".join(html)
    final_html = inject_floating_toc(final_html)
    return final_html


def save_html(html_text: str, out_html_path: str) -> None:
    with open(out_html_path, "w", encoding="utf-8") as f:
        f.write(html_text)