# Regression suite — لازم يعدّي قبل أي deploy

## التشغيل

```bash
pip install -r requirements.txt -r requirements-dev.txt
python -m pytest tests -q                       # سريع، من غير نت ولا OpenRouter (~2 ثانية)
OPENROUTER_API_KEY=sk-or-... python evals/run_understanding_eval.py --repeat 3   # الموديل الحقيقي
```

`deploy.sh` بيشغّل `tests/` على الكود الجديد **قبل** ما يلمس السيرفر، ولو فيه test واحد فشل بيوقف الـ deploy.

## فيه إيه

| الملف | بيختبر إيه |
|---|---|
| `test_production_scenarios.py` | المحادثات اللي فشلت في production بالظبط (حلقة التحويل، الأزمة النفسية، الإلغاء الغلط، "الدكتور نفسي"، اللهجة المصرية)، شغالة end-to-end من نفس الـ entry point اللي n8n بيكلمه |
| `test_gates.py` | كل gate لوحده: التحويل، الإلغاء، اسم الدكتور مقابل التخصص، الـ routing، وتحليل رد الـ LLM |
| `evals/` | الموديل الحقيقي على جمل **جديدة** بلهجات مختلفة. ده اللي بيثبت إن الفهم مش معتمد على كلمات محفوظة |

## القاعدة

أي حادثة جديدة في production:
1. تتضاف كـ case في `evals/understanding_cases.json` لو كانت مشكلة فهم.
2. أو تتضاف كـ test في `test_production_scenarios.py` لو كانت مشكلة wiring (الموديل فهم صح بس الكود عمل حاجة غلط).
3. **ممنوع** تتصلّح بـ regex جديد أو verifier جديد من غير test بيوضح إنه مش بيكسر السيناريوهات اللي هنا.

## ليه الـ tests دي مش بتـ script الموديل في flows طويلة

الـ tests القديمة (`test_agent_graph.py`، `test_app_http.py`) كانت فاشلة **قبل** التعديلات دي. السبب إن الـ verifiers بيعيدوا كتابة الرد المتـ script وبيستهلكوا الردود اللي بعده، وده نفس تعارض الـ guards اللي بيحصل في production. عشان كده الـ flows اللي فيها عمليات مش بترجع (زي الإلغاء) بتتختبر على مستوى الـ tool، بـ state مبني من المحادثة الحقيقية.
