# 1. create a monitoring namespace
kubectl create namespace monitoring

# 2. add helm repos and update
helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
helm repo add grafana https://grafana.github.io/helm-charts
helm repo update

# 3. (optional) create a values file (see example below) and then install
helm install kube-prom-stack prometheus-community/kube-prometheus-stack \
  --namespace monitoring \
  -f ./kube-prom-values.yaml

# 4. check status
kubectl -n monitoring get pods

# 5. access Grafana (quick access via port-forward)
kubectl -n monitoring port-forward svc/kube-prom-stack-grafana 3000:80
# then open http://localhost:3000
# default user usually "admin" — password can be obtained or set via values
kubectl -n monitoring get secret kube-prom-stack-grafana -o jsonpath="{.data.admin-password}" | base64 --decode ; echo

