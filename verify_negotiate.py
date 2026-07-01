import urllib.request

with urllib.request.urlopen("http://localhost:8080/") as r:
    html = r.read().decode("utf-8")

btn       = html.find('id="getNegotiateBtn"')
qa_div    = html.find('id="panel-qa"')
risks_div = html.find('id="panel-risks"')
neg       = html.find('id="negotiateFooter"')
css       = html.find("risk-panel-footer")
jsref     = html.find("const negotiateFooter")
dl        = html.find('id="downloadReportBtn"')
fn        = html.find("doNegotiate")
margin    = html.find("margin-top: 24px")

print("id=panel-risks div:   ", risks_div)
print("id=getNegotiateBtn:   ", btn)
print("id=panel-qa div:      ", qa_div)
print("id=downloadReportBtn: ", dl)
print("doNegotiate func:     ", fn)
print("margin-top:24px:      ", margin)
print("id=negotiateFooter:   ", neg)
print("risk-panel-footer CSS:", css)
print("const negotiateFooter:", jsref)
print()

ok_inside  = risks_div < btn < qa_div
ok_no_neg  = neg == -1
ok_no_css  = css == -1
ok_no_jsref = jsref == -1
ok_dl      = dl != -1

print("PASS" if ok_inside  else "FAIL", "Button is inside #panel-risks (risks_div < btn < qa_div)")
print("PASS" if ok_no_neg  else "FAIL", "negotiateFooter div removed")
print("PASS" if ok_no_css  else "FAIL", "risk-panel-footer CSS removed")
print("PASS" if ok_no_jsref else "FAIL", "negotiateFooter JS ref removed")
print("PASS" if ok_dl      else "FAIL", "Download Report button present")
print()
print("ALL PASS" if all([ok_inside, ok_no_neg, ok_no_css, ok_no_jsref, ok_dl]) else "SOME FAILURES")
