# Hard negatives: pasted JSON / YAML / config / log snippets with a question.
ROWS = [
dict(cat="hn_config_paste", lang="en", sys=None, prompt="""Here is part of my package.json. Why does npm still install a 2.x release of the logger?

"dependencies": {
  "fastify": "^4.26.0",
  "pino": "^2.14.1",
  "zod": "~3.22.4"
}"""),
dict(cat="hn_config_paste", lang="en", sys=None, prompt="""What is wrong with this docker-compose fragment? The app container cannot reach the database.

services:
  app:
    image: myapp:latest
    ports: ["3000:3000"]
  db:
    image: postgres:16
    environment:
      POSTGRES_PASSWORD: secret"""),
dict(cat="hn_config_paste", lang="en", sys="You are a careful systems engineer. Explain before you prescribe.", prompt="""Read this nginx block and tell me why requests to /api/ lose their path prefix.

location /api/ {
  proxy_pass http://backend/;
  proxy_set_header Host $host;
}"""),
dict(cat="hn_config_paste", lang="en", sys=None, prompt="""This log line repeats every few seconds. What is the service actually complaining about?

WARN [pool-2-thread-9] HikariPool-1 - Connection is not available, request timed out after 30000ms (total=10, active=10, idle=0, waiting=14)"""),
dict(cat="hn_config_paste", lang="en", sys=None, prompt="""Explain what this systemd unit does on boot and whether Restart=always is a good idea here.

[Service]
ExecStart=/usr/local/bin/sync-daemon --once
Restart=always
RestartSec=1"""),
dict(cat="hn_config_paste", lang="en", sys=None, prompt="""Here is a Kubernetes probe config. The pod keeps getting killed during startup. What would you change?

livenessProbe:
  httpGet: {path: /healthz, port: 8080}
  initialDelaySeconds: 2
  periodSeconds: 5
  failureThreshold: 2"""),
dict(cat="hn_config_paste", lang="en", sys=None, prompt="""Can you walk me through what this .gitignore actually excludes? I think it is too broad.

build/
*.log
!important.log
/tmp
config/*.local.*"""),
dict(cat="hn_config_paste", lang="en", sys=None, prompt="""What does this Cargo.toml profile do to build times and binary size?

[profile.release]
lto = "fat"
codegen-units = 1
panic = "abort"
strip = "symbols\""""),
dict(cat="hn_config_paste", lang="en", sys=None, prompt="""My CI fails on this step with 'permission denied'. What is it trying to do?

- name: Publish
  run: |
    chmod +x ./scripts/release.sh
    ./scripts/release.sh --tag "$GITHUB_REF_NAME\""""),
dict(cat="hn_config_paste", lang="en", sys="You answer support tickets for a hosting company. Be concrete.", prompt="""A customer sent this .env and says mail is not going out. Anything obviously off?

SMTP_HOST=smtp.example.com
SMTP_PORT=465
SMTP_TLS=starttls
SMTP_USER=noreply@example.com"""),
dict(cat="hn_config_paste", lang="en", sys=None, prompt="""Explain what this JSON response tells me about the failure. The client shows a blank page.

{"status": 502, "error": "Bad Gateway", "upstream": "api-7f4c9", "retryable": true, "request_id": "01HQ..."}"""),
dict(cat="hn_config_paste", lang="en", sys=None, prompt="""Here is a stack trace tail. Where would you start looking?

ValueError: could not convert string to float: ''
  File "etl/clean.py", line 88, in normalise
  File "etl/clean.py", line 41, in run
  File "main.py", line 12, in <module>"""),
dict(cat="hn_config_paste", lang="en", sys=None, prompt="""What is this Terraform block going to do on the next apply, given the bucket already exists?

resource "aws_s3_bucket" "logs" {
  bucket        = "acme-logs"
  force_destroy = true
}"""),
dict(cat="hn_config_paste", lang="en", sys=None, prompt="""Tell me what this crontab line runs and when, and whether the output goes anywhere.

*/15 6-20 * * 1-5 /opt/etl/run.sh >> /var/log/etl.log 2>&1"""),
dict(cat="hn_config_paste", lang="en", sys=None, prompt="""Is anything unsafe in this Postgres connection string as written?

postgres://app:hunter2@db.internal:5432/orders?sslmode=disable&pool_max_conns=200"""),
dict(cat="hn_config_paste", lang="en", sys=None, prompt="""Here is an eslint config a colleague added. What behaviour changes for the whole repo?

{"extends": ["eslint:recommended"], "rules": {"no-unused-vars": "off", "eqeqeq": ["error", "smart"]}}"""),
dict(cat="hn_config_paste", lang="en", sys=None, prompt="""Explain the difference these two YAML values make. I cannot tell if the second is a string or a boolean.

flags:
  strict: yes
  verbose: "no\""""),
dict(cat="hn_config_paste", lang="en", sys=None, prompt="""This Grafana alert fires constantly overnight. Reading the rule, why might that be?

expr: rate(http_requests_total{code=~"5.."}[1m]) > 0
for: 30s"""),
dict(cat="hn_config_paste", lang="en", sys=None, prompt="""What is this SQL EXPLAIN telling me about the join?

Nested Loop  (cost=0.43..91422.10 rows=1 width=48)
  ->  Seq Scan on orders o  (rows=1204331)
  ->  Index Scan using customers_pkey on customers c"""),
dict(cat="hn_config_paste", lang="en", sys=None, prompt="""Here is the tail of a failed deploy. In plain words, what went wrong?

Step 7/9 : COPY --from=build /app/dist ./dist
COPY failed: stat /var/lib/docker/tmp/.../app/dist: no such file or directory"""),
dict(cat="hn_config_paste", lang="en", sys=None, prompt="""Read this tsconfig and tell me why my imports of .js files from TypeScript keep failing.

{"compilerOptions": {"module": "commonjs", "moduleResolution": "node16", "allowJs": false, "strict": true}}"""),
dict(cat="hn_config_paste", lang="en", sys=None, prompt="""What does this rate limit header set mean for a client that just got a 429?

X-RateLimit-Limit: 600
X-RateLimit-Remaining: 0
X-RateLimit-Reset: 1714500000
Retry-After: 43"""),
dict(cat="hn_config_paste", lang="en", sys=None, prompt="""Here is a fragment of an Ansible play. Will the handler actually run if the template is unchanged?

- template:
    src: app.conf.j2
    dest: /etc/app.conf
  notify: restart app"""),
dict(cat="hn_config_paste", lang="en", sys="You review infrastructure changes. Flag risk, then explain.", prompt="""Anything I should worry about in this firewall rule before I merge it?

-A INPUT -p tcp --dport 5432 -s 0.0.0.0/0 -j ACCEPT"""),
dict(cat="hn_config_paste", lang="en", sys=None, prompt="""Explain what this webpack alias does and why one import still resolves to node_modules.

resolve: {
  alias: {"@lib": path.resolve(__dirname, "src/lib")},
  modules: ["node_modules", "src"]
}"""),
dict(cat="hn_config_paste", lang="en", sys=None, prompt="""This appears in our audit log. What sequence of events does it describe?

12:04:11 user=amelia action=role.grant target=svc-billing role=admin source=console
12:04:19 user=amelia action=key.create target=svc-billing ttl=none"""),
dict(cat="hn_config_paste", lang="en", sys=None, prompt="""What is the practical effect of these two cache headers together?

Cache-Control: public, max-age=0, must-revalidate
ETag: W/"9f1c-k2\""""),
dict(cat="hn_config_paste", lang="en", sys=None, prompt="""My colleague wrote this Makefile target and it reruns every time. Why?

report.pdf:
	pandoc report.md -o report.pdf
.PHONY: report.pdf"""),
dict(cat="hn_config_paste", lang="en", sys=None, prompt="""Here is a snippet of a CSV export. Why would a spreadsheet show the last column as a date?

sku,qty,ratio
AB-19,4,3-5
AB-20,1,10-12"""),
dict(cat="hn_config_paste", lang="en", sys=None, prompt="""Explain what this JWT payload allows, assuming the signature checks out.

{"sub": "u_8812", "scope": "read:orders write:orders", "exp": 1899999999, "aud": "internal"}"""),
dict(cat="hn_config_paste", lang="en", sys=None, prompt="""What does this pip resolver message mean in practice, and what are my options?

ERROR: Cannot install app==2.1.0 and numpy==1.24.0 because these package versions have conflicting dependencies."""),
dict(cat="hn_config_paste", lang="en", sys=None, prompt="""Read this retry policy and tell me the worst-case total wait before the caller sees an error.

{"attempts": 5, "initial_ms": 200, "multiplier": 3, "jitter": false, "timeout_ms": 2000}"""),
dict(cat="hn_config_paste", lang="en", sys=None, prompt="""Here is a Prometheus scrape config. Why might half the targets be missing labels?

- job_name: nodes
  kubernetes_sd_configs: [{role: pod}]
  relabel_configs:
    - source_labels: [__meta_kubernetes_pod_label_app]
      target_label: app"""),
dict(cat="hn_config_paste", lang="en", sys=None, prompt="""What is this git config doing to my line endings on a Linux machine?

[core]
	autocrlf = true
	safecrlf = warn"""),
dict(cat="hn_config_paste", lang="en", sys=None, prompt="""Explain this browser console error in ordinary language, and what usually causes it.

Access to fetch at 'https://api.example.com/v1/me' from origin 'https://app.example.com' has been blocked by CORS policy: No 'Access-Control-Allow-Origin' header is present."""),
dict(cat="hn_config_paste", lang="en", sys=None, prompt="""Given this Redis INFO excerpt, is memory the problem or is something else going on?

used_memory_human:5.98G
maxmemory_human:6.00G
evicted_keys:0
blocked_clients:37"""),
dict(cat="hn_config_paste", lang="de", sys=None, prompt="""Was macht dieser Abschnitt aus unserer Konfiguration genau? Die Zeitzone stimmt in den Berichten nicht.

report:
  timezone: UTC
  day_start: "00:00"
  locale: de-DE"""),
dict(cat="hn_config_paste", lang="de", sys=None, prompt="""Diese Zeile steht seit gestern im Log. Was sagt sie aus?

ERROR c.a.JobRunner - job=nightly-export status=FAILED attempt=3/3 cause=SocketTimeoutException after 60s"""),
dict(cat="hn_config_paste", lang="fr", sys=None, prompt="""Explique ce que fait cette configuration et pourquoi les fichiers depassent la taille limite.

logging:
  rotate: daily
  max_size: 500MB
  keep: 30
  compress: false"""),
dict(cat="hn_config_paste", lang="fr", sys=None, prompt="""Voici une reponse de notre API. Que signifie ce champ pour le client qui l'appelle?

{"ok": false, "code": "quota_exceeded", "reset_at": "2026-03-01T00:00:00Z", "grace": 0}"""),
dict(cat="hn_config_paste", lang="es", sys=None, prompt="""Que hace exactamente esta regla de despliegue? Los cambios tardan horas en verse.

cdn:
  ttl: 86400
  stale_while_revalidate: 0
  purge_on_deploy: false"""),
dict(cat="hn_config_paste", lang="es", sys=None, prompt="""Puedes explicarme este error de la base de datos y que suele provocarlo?

ERROR: deadlock detected
DETAIL: Process 4412 waits for ShareLock on transaction 99182; blocked by process 4398."""),
dict(cat="hn_config_paste", lang="it", sys=None, prompt="""Cosa fa questa parte del file di configurazione? I backup occupano troppo spazio.

backup:
  schedule: hourly
  retention_days: 90
  incremental: false"""),
dict(cat="hn_config_paste", lang="pt", sys=None, prompt="""O que esta linha de log indica sobre a fila de mensagens?

WARN rabbit consumer=orders prefetch=1 unacked=1 queue_depth=41822 idle_ms=0"""),
dict(cat="hn_config_paste", lang="nl", sys=None, prompt="""Wat doet dit stukje configuratie precies? Gebruikers worden elke ochtend uitgelogd.

session:
  ttl_minutes: 480
  sliding: false
  absolute_timeout: true"""),
dict(cat="hn_config_paste", lang="ja", sys=None, prompt="""この設定の意味を教えてください。夜間のバッチが途中で止まります。

worker:
  concurrency: 8
  timeout_seconds: 300
  retry: 0"""),
dict(cat="hn_config_paste", lang="ja", sys=None, prompt="""次のログは何を示していますか。原因の見当をつけたいです。

ERROR sync failed: remote closed connection after 120s, 0 of 4812 records written"""),
dict(cat="hn_config_paste", lang="ko", sys=None, prompt="""이 설정이 어떤 동작을 하는지 설명해 주세요. 알림이 두 번씩 갑니다.

notifications:
  channels: [email, push]
  dedupe_window_seconds: 0"""),
dict(cat="hn_config_paste", lang="zh", sys=None, prompt="""这段配置是什么意思？上传大文件时总是失败。

upload:
  max_body_size: 8MB
  timeout: 30s
  chunked: false"""),
dict(cat="hn_config_paste", lang="is", sys=None, prompt="""Hvad gerir thessi stilling nakvaemlega? Skyrslan kemur alltaf tom a mánudogum.

schedule:
  cron: "0 6 * * 1"
  window_days: 0
  include_weekend: false"""),
]
