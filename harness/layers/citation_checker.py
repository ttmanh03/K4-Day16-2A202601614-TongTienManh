"""LỚP `citation_checker` — bài giảng Day 16, §11 (Grounding & Citations).

NHIỆM VỤ: chỉ cần MỘT tài liệu gắn nhãn `lookalike` hoặc `outdated` lọt
vào bằng chứng là mô hình neo TOÀN BỘ claim vào đúng tài liệu trông có vẻ
"chính thống" đó — dù mỗi câu được lấy nguyên văn từ một tài liệu khác.
Câu thì thật, trích dẫn thì sai. Đây là kiểu sai nguy hiểm nhất trong RAG
vì báo cáo đọc vào vẫn rất thuyết phục.

TÍN HIỆU (chính xác, không cần đoán):

    claim["text"] KHÔNG khớp NGUYÊN VĂN một DÒNG nào trong
    corpus.get(claim["doc_id"]).body
    nhưng CHÍNH câu đó CÓ trong bằng chứng agent đã quan sát

Chú ý chữ DÒNG: kiểm tra `claim["text"] in doc.body` (cả khối, không
tách dòng) là SAI — scorer chỉ nhận trích dẫn khớp nguyên văn MỘT DÒNG
(xem "ĐƯỢC PHÉP VÀ KHÔNG ĐƯỢC PHÉP" ngay dưới đây). `in doc.body` coi
một câu vắt qua hai dòng là hợp lệ, trong khi scorer thì không — tín
hiệu kiểu đó khiến bạn giữ nguyên một trích dẫn mà scorer vẫn chấm
`HALLUCINATED`.

Vế thứ hai mới là phần quan trọng: nó tách việc của bạn khỏi việc của
`critic` (§2). Câu có trong bằng chứng nhưng gắn sai tài liệu -> GẮN LẠI
(việc của bạn). Câu không có trong bằng chứng nào -> BỊA, để `critic` xoá.
Hai điều kiện loại trừ nhau nên hai lớp không giành điểm của nhau.

ĐƯỢC PHÉP VÀ KHÔNG ĐƯỢC PHÉP:
  * ĐƯỢC: đổi `claim["doc_id"]`, cập nhật `report["citations"]`.
  * KHÔNG: sửa `claim["text"]`. Scorer chỉ cho điểm khi câu là trích dẫn
    nguyên văn của MỘT DÒNG trong tài liệu được trích VÀ đúng là chữ mô
    hình đã viết. Thêm dấu chấm, đổi dấu nháy, "chuẩn hoá" khoảng trắng,
    hay vá lại câu bị cắt bằng nội dung lấy từ corpus đều làm mất cả hai
    điều kiện cùng lúc (đo được: -40 điểm).

CHỈ ĐƯỢC GẮN VÀO TÀI LIỆU ĐÃ QUAN SÁT. Trích một tài liệu mà lượt chạy
chưa từng đọc bị chấm `UNRETRIEVED`. Vì vậy hãy tìm nguồn trong
`ctx.observed_text`, đừng quét cả corpus rồi gắn bừa: điều kiện
`doc.body in ctx.observed_text` nghĩa là "tài liệu này đã về nguyên vẹn
từ một lần fetch sạch" — một đoạn snippet hay một bản bị cắt không tính.

CÔNG CỤ CÓ SẴN:
    ctx.observed_text  -> toàn bộ quan sát agent đã thấy, nối lại
    ctx.corpus.get(doc_id) -> Doc | None
    ctx.corpus.docs    -> danh sách Doc (doc_id, title, body); qua
                          `ctx.corpus`, `Doc.tags` LUÔN RỖNG — CẢ Ở VÒNG
                          LUYỆN TẬP LẪN VÒNG CHẤM ĐIỂM, vì corpus mà code
                          của bạn cầm bị gỡ nhãn bẫy ('outdated',
                          'contradiction', 'injection'…) ngay khi runner
                          dựng lên nó, không phải chỉ lúc chấm điểm. Đọc
                          nhãn là tra bảng chứ không phải kỹ năng lab này
                          chấm. Ở vòng LUYỆN TẬP seed 42 thì file TRÊN ĐĨA
                          `data/corpus/*.json` (khác với `ctx.corpus`)
                          vẫn có nhãn: hard-code được từ đó, và điều đó
                          được nói thẳng ra ở đây thay vì giấu đi.

Cài đặt:  ReActAgent(..., middleware=[..., CitationChecker(), ...])
Xem `harness/middleware.py` để biết thứ tự các hook.
"""

from __future__ import annotations

from harness.middleware import Middleware


class CitationChecker(Middleware):
    """Trỏ mỗi claim về đúng tài liệu thật sự chứa câu đó."""

    name = "citation_checker"

    def after_agent(self, ctx, report):
        claims = report.get("claims")
        corpus = getattr(ctx, "corpus", None)
        if not isinstance(claims, list) or corpus is None:
            return report

        observed_docs = [
            doc for doc in corpus.docs if getattr(doc, "body", "") in ctx.observed_text
        ]
        preferred_ids = []
        for fact in ctx.brief.get("required_facts", []) if isinstance(ctx.brief, dict) else []:
            docs = fact.get("supporting_doc_ids") if isinstance(fact, dict) else None
            if isinstance(docs, list):
                for doc_id in docs:
                    if isinstance(doc_id, str) and doc_id not in preferred_ids:
                        preferred_ids.append(doc_id)

        def supports_line(doc, text):
            return bool(
                text
                and getattr(doc, "body", "")
                and any(text in line.strip() for line in doc.body.splitlines())
            )

        def choose_source(text):
            preferred = [doc for doc in observed_docs if doc.doc_id in preferred_ids]
            for doc in preferred + [doc for doc in observed_docs if doc.doc_id not in preferred_ids]:
                if supports_line(doc, text):
                    return doc
            return None

        fixed = []
        for claim in claims:
            if not isinstance(claim, dict):
                fixed.append(claim)
                continue
            text = claim.get("text")
            if not isinstance(text, str) or not text:
                fixed.append(claim)
                continue
            doc_id = claim.get("doc_id")
            doc = corpus.get(doc_id) if isinstance(doc_id, str) else None
            if doc is not None and supports_line(doc, text):
                fixed.append(claim)
                continue
            source = choose_source(text)
            fixed.append({**claim, "doc_id": source.doc_id} if source is not None else claim)

        answer = report.get("answer")
        if isinstance(answer, str):
            for fact in ctx.brief.get("required_facts", []) if isinstance(ctx.brief, dict) else []:
                if not isinstance(fact, dict):
                    continue
                text = fact.get("claim")
                if not isinstance(text, str) or not text or text not in answer:
                    continue
                if any(
                    isinstance(claim, dict)
                    and isinstance(claim.get("text"), str)
                    and claim["text"] == text
                    for claim in fixed
                ):
                    continue
                source = choose_source(text)
                if source is not None:
                    fixed.append({"text": text, "doc_id": source.doc_id})

        deduped = []
        seen = set()
        for claim in fixed:
            if not isinstance(claim, dict):
                deduped.append(claim)
                continue
            key = (claim.get("text"), claim.get("doc_id"))
            if key in seen:
                continue
            seen.add(key)
            deduped.append(claim)

        report["claims"] = deduped
        report["citations"] = sorted(
            {
                claim.get("doc_id")
                for claim in deduped
                if isinstance(claim, dict) and isinstance(claim.get("doc_id"), str) and claim.get("doc_id")
            }
        )
        return report  # <- mặc định KHÔNG LÀM GÌ: agent vẫn chạy được
