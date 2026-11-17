# Extraction quality: 10 graph triples graded by hand

**Counts: correct 3 / wrong 5 / unsupported 2**

Sample: `python -m app.graph sample --n 10 --seed 42`, 10 of 49 edges of the resolved graph (rules: identifier,normalised,alias; chunks 800/10). Graph hash `a461fb5b66bf96f8`, fingerprint `359e5c0282635100`, build `20261117-171907-137697`.
Rules: `data/results/v3/graph_triples_grading_rules.md`. Graded by: Muhammad Saad JAVED. Reviewed by: Sultan Amiour.
Ten triples are a sanity check, not an accuracy measure: one triple moves a rate by 10 points.

| # | Triple | Evidence chunk(s) | Quote | Grade | Graded on | Note | Review |
|--:|---|---|---|---|---|---|---|
| 1 | KST-2044 (ErrorCode) -**escalates_to**-> Data Integrations (Team) | `support-runbook-errors_md:4` | "to Data Integrations if several customers report it for the same source type at the same time" | wrong | support-runbook-errors_md:4 | The Escalate line belongs to KST-2040 (its section is cut at the chunk start); KST-2044's own section says 'Escalate: no'. |  |
| 2 | Invoice (Product) -**has_feature**-> company details (Feature) | `billing-policy-2026_md:2` | "The billing contact and the company details printed on invoices (name, address, VAT number)" | wrong | billing-policy-2026_md:2 | The chunk says company details are printed on invoices, not that an invoice offers a feature: wrong relation. |  |
| 3 | Invoices (Product) -**has_feature**-> company details (Feature) | `billing-policy-2025_md:2` | "Invoices can also be downloaded from Settings → Plan → Invoices. The billing contact and the company details printed on invoices (name, address, VAT number) can be changed by the Owner; changes apply to future invoices only." | wrong | billing-policy-2025_md:2 | Same as #2 (2025 policy): printed on invoices is not has_feature. 'Invoice' and 'Invoices' were not merged (no plural rule). |  |
| 4 | dataset (Product) -**has_feature**-> schedule (Feature) | `product-guide_md:8` | "Any dashboard can be sent as a PDF on a schedule" | wrong | product-guide_md:8 | The quote says dashboards are sent on a schedule; the chunk only gives datasets a refresh frequency: wrong entity. |  |
| 5 | Kestrel (Product) -**has_feature**-> Forgot password (Feature) | `product-guide_md:7` | "To reset a forgotten password, choose **Forgot password** on the sign-in page." | unsupported | product-guide_md:7 | Product guide: 'choose Forgot password on the sign-in page' (Kestrel's sign-in). Judgment call: the sentence does not name Kestrel. | Grader: correct; reviewer: unsupported. The chunk never names Kestrel, so the link is inferred; rules: only the evidence chunk counts, pick the worse grade when unsure. Settled: unsupported. |
| 6 | single sign-on (Feature) -**included_in**-> Enterprise (Plan) | `product-guide_md:5` | "On the Business and Enterprise plans, workspaces can use single sign-on with any SAML 2.0 identity provider." | correct | product-guide_md:5 | 'On the Business and Enterprise plans, workspaces can use single sign-on.' |  |
| 7 | Kestrel Analytics (Product) -**has_feature**-> dashboard (Feature) | `product-guide_md:0` | "turn those datasets into dashboards that refresh automatically" | correct | product-guide_md:0 | 'Kestrel Analytics is a dashboard and business intelligence service'; dashboards are built from datasets. |  |
| 8 | Owner (Product) -**has_feature**-> billing contact (Feature) | `billing-policy-2026_md:2` | "The billing contact and the company details printed on invoices (name, address, VAT number) can be changed by the Owner; changes apply to future invoices only." | wrong | billing-policy-2026_md:2 | The Owner can change the billing contact; Owner is a role (typed Product), not something that offers a feature: wrong entity and relation. |  |
| 9 | single sign-on (Feature) -**causes**-> KST-1001 (ErrorCode) | `support-runbook-errors_md:1` | "wrong password, or the user is signing in to a workspace that enforces single sign-on" | correct | support-runbook-errors_md:1 | Cause of KST-1001: signing in to a workspace that enforces single sign-on; direction cause -> error is right. | Reviewer agreed: correct; the cause is signing in to a workspace that enforces single sign-on (a condition). |
| 10 | Owner (Product) -**has_feature**-> Settings → Plan → Invoices (Feature) | `billing-policy-2026_md:2` | "Invoices can also be downloaded from **Settings → Plan → Invoices**." | unsupported | billing-policy-2026_md:2 | The quote only says invoices are downloaded from that menu; nothing in the chunk links the Owner to it. | Reviewer agreed: unsupported; Owner is a role typed as Product, same issue as #8. |

## Chunk excerpts

**1. KST-2044 -escalates_to-> Data Integrations**
- `support-runbook-errors_md:4`: ps the connection. / **Support action:** ask the customer to confirm the database is reachable from outside their network, on the host and port configured in the connector. / **Escalate:** to Data Integrations if several customers report it for the same source type at the same time. /  / ### KST-2044 — Source refused the connection /  / **User sees:** "Your data source refused the connection from Kestrel." / **Cause:** the database only accepts connections from an allowlist of IP addresses, and the Kestrel ...

**2. Invoice -has_feature-> company details**
- `billing-policy-2026_md:2`: ... are held by our payment processor. /  / ## 4. Invoices /  / An invoice is issued for every charge and sent by e-mail to the billing contact of the workspace. Invoices can also be downloaded from **Settings → Plan → Invoices**. The billing contact and the company details printed on invoices (name, address, VAT number) can be changed by the Owner; changes apply to future invoices only. A corrected copy of a past invoice can be requested from support within 12 months of its date. /  / ## 5. Adding and removing seats /  / Seats can be added at ...

**3. Invoices -has_feature-> company details**
- `billing-policy-2025_md:2`:  We do not store full card numbers; they are held by our payment processor. /  / ## 4. Invoices /  / An invoice is issued for every charge and sent by e-mail to the billing contact of the workspace. Invoices can also be downloaded from **Settings → Plan → Invoices**. The billing contact and the company details printed on invoices (name, address, VAT number) can be changed by the Owner; changes apply to future invoices only. A corrected copy of ...

**4. dataset -has_feature-> schedule**
- `product-guide_md:8`: ...ection 2). /  / You can see when a dataset was last refreshed by hovering over the clock icon in the corner of any tile. A red clock means the last refresh failed; the tooltip shows the error code. /  / ## 8. Scheduled reports /  / Any dashboard can be sent as a PDF on a schedule: daily, weekly or monthly. A schedule can have up to 50 recipients, wh

**5. Kestrel -has_feature-> Forgot password**
- `product-guide_md:7`: port verifies your identity before resetting anything; this verification can take up to 2 business days. /  / To reset a forgotten password, choose **Forgot password** on the sign-in page. The reset link is valid for 1 hour. /  / ## 6. Data connectors /  / A connector stores the connection details for one data source. Kestrel supports PostgreSQL, MySQL, Google BigQuery, Snowflake, Google Sheets and CSV upload. A ...

**6. single sign-on -included_in-> Enterprise** (aliases: SSO, Single sign-on (SAML), Enterprise plan, Enterprise plans)
- `product-guide_md:5`: ...authenticator device. /  / Admins can make two-factor authentication mandatory for every member from **Settings → Security**. Members who have not set it up are asked to do so at their next sign-in. /  / ### 5.2 Single sign-on /  / On the Business and Enterprise plans, workspaces can use single sign-on with any SAML 2.0 identity provider. To set it up, an Admin opens **Settings → Security → Single sign-on**, downloads the Kestrel service provider metadata, registers Kestrel in the identity provider, and pastes the identity provider metadata URL back into...

**7. Kestrel Analytics -has_feature-> dashboard**
- `product-guide_md:0`: ... Audience: customers and workspace administrators /  / ## 1. What Kestrel does /  / Kestrel Analytics is a dashboard and business intelligence service for teams. You connect your data sources, build datasets on top of them, and turn those datasets into dashboards that refresh automatically. Dashboards can be shared with colleagues, sent as scheduled PDF reports, or embedded in your own web applications. /  / Everything in Kestrel lives inside a **workspace**. A workspace has members, connectors, datasets and ...

**8. Owner -has_feature-> billing contact**
- `billing-policy-2026_md:2`: ... are held by our payment processor. /  / ## 4. Invoices /  / An invoice is issued for every charge and sent by e-mail to the billing contact of the workspace. Invoices can also be downloaded from **Settings → Plan → Invoices**. The billing contact and the company details printed on invoices (name, address, VAT number) can be changed by the Owner; changes apply to future invoices only. A corrected copy of a past invoice can be requested from support within 12 months of its date. /  / ## 5. Adding and removing seats /  / Seats can be added at any time

**9. single sign-on -causes-> KST-1001** (aliases: SSO, Single sign-on (SAML))
- `support-runbook-errors_md:1`: ...ur with a workaround | Next business day | / | SEV-4 | Question, cosmetic issue | Normal queue | /  / ## Sign-in (KST-1xxx) /  / ### KST-1001 — Invalid credentials /  / **User sees:** "The e-mail or password is incorrect." / **Cause:** wrong password, or the user is signing in to a workspace that enforces single sign-on. / **Support action:** check whether the workspace enforces single sign-on. If it 

**10. Owner -has_feature-> Settings → Plan → Invoices**
- `billing-policy-2026_md:2`: ... Payment method**. We do not store full card or account numbers; they are held by our payment processor. /  / ## 4. Invoices /  / An invoice is issued for every charge and sent by e-mail to the billing contact of the workspace. Invoices can also be downloaded from **Settings → Plan → Invoices**. The billing contact and the company details printed on invoices (name, address, VAT number) can be changed by the Owner; changes apply to future invoices only. A corrected copy of a past invoice can be requested from su...

