package huggingface

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

func testClient(t *testing.T, handler http.HandlerFunc) *Client {
	t.Helper()
	server := httptest.NewServer(handler)
	t.Cleanup(server.Close)
	namespace, token := "test-namespace", "test-token"
	client, err := NewClient(&server.URL, &namespace, &token)
	if err != nil {
		t.Fatal(err)
	}
	return client
}

func TestClientRoutes(t *testing.T) {
	tests := []struct {
		name     string
		method   string
		path     string
		response string
		call     func(*Client) error
	}{
		{
			name:     "list endpoints",
			method:   http.MethodGet,
			path:     "/test-namespace",
			response: `{"items":[]}`,
			call: func(client *Client) error {
				_, err := client.ListEndpoints()
				return err
			},
		},
		{
			name:     "create endpoint",
			method:   http.MethodPost,
			path:     "/test-namespace",
			response: `{"name":"example"}`,
			call: func(client *Client) error {
				_, err := client.CreateEndpoint(CreateEndpointRequest{Name: "example"})
				return err
			},
		},
		{
			name:     "delete endpoint",
			method:   http.MethodDelete,
			path:     "/test-namespace/example",
			response: "",
			call: func(client *Client) error {
				return client.DeleteEndpoint("example")
			},
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			client := testClient(t, func(w http.ResponseWriter, r *http.Request) {
				if r.Method != test.method || r.URL.Path != test.path {
					t.Errorf("unexpected request: %s %s", r.Method, r.URL.Path)
					w.WriteHeader(http.StatusBadRequest)
					return
				}
				_, _ = w.Write([]byte(test.response))
			})

			if err := test.call(client); err != nil {
				t.Fatal(err)
			}
		})
	}
}

func TestGetEndpointParsesStatusURLAndLastUsedAt(t *testing.T) {
	client := testClient(t, func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodGet || r.URL.Path != "/test-namespace/example" {
			t.Errorf("unexpected request: %s %s", r.Method, r.URL.Path)
			w.WriteHeader(http.StatusBadRequest)
			return
		}
		_, _ = w.Write([]byte(`{"name":"example","status":{"url":"https://example.endpoints.huggingface.cloud","lastUsedAt":"2026-07-31T13:41:02Z"}}`))
	})

	endpoint, err := client.GetEndpoint("example")
	if err != nil {
		t.Fatal(err)
	}
	if endpoint.Status.URL != "https://example.endpoints.huggingface.cloud" {
		t.Fatalf("got URL %q", endpoint.Status.URL)
	}
	if endpoint.Status.LastUsedAt != "2026-07-31T13:41:02Z" {
		t.Fatalf("got LastUsedAt %q", endpoint.Status.LastUsedAt)
	}
}

func TestUpdateEndpointMarshalsProvider(t *testing.T) {
	client := testClient(t, func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPut || r.URL.Path != "/test-namespace/example" {
			t.Errorf("unexpected request: %s %s", r.Method, r.URL.Path)
			w.WriteHeader(http.StatusBadRequest)
			return
		}
		var request UpdateEndpointRequest
		if err := json.NewDecoder(r.Body).Decode(&request); err != nil {
			t.Error(err)
			w.WriteHeader(http.StatusBadRequest)
			return
		}
		if request.Provider == nil || request.Provider.Region != "us-east-1" || request.Provider.Vendor != "aws" {
			t.Errorf("got provider %#v", request.Provider)
			w.WriteHeader(http.StatusBadRequest)
			return
		}
		_, _ = w.Write([]byte(`{"name":"example"}`))
	})

	_, err := client.UpdateEndpoint("example", UpdateEndpointRequest{
		Provider: &Provider{Region: "us-east-1", Vendor: "aws"},
	})
	if err != nil {
		t.Fatal(err)
	}
}

func TestHealthRouteMarshalsCamelCase(t *testing.T) {
	healthRoute := "/health"
	images := []struct {
		name  string
		image any
	}{
		{"tei", Tei{HealthRoute: &healthRoute}},
		{"llamacpp", Llamacpp{HealthRoute: &healthRoute}},
		{"tgi-neuron", TgiNeuron{HealthRoute: &healthRoute}},
		{"tgi", Tgi{HealthRoute: &healthRoute}},
		{"custom", Custom{HealthRoute: &healthRoute}},
		{"vllm", Vllm{HealthRoute: &healthRoute}},
	}

	for _, test := range images {
		t.Run(test.name, func(t *testing.T) {
			body, err := json.Marshal(test.image)
			if err != nil {
				t.Fatal(err)
			}
			if !strings.Contains(string(body), `"healthRoute":"/health"`) {
				t.Fatalf("expected healthRoute in %s", body)
			}
			if strings.Contains(string(body), "health_route") {
				t.Fatalf("unexpected health_route in %s", body)
			}
		})
	}
}

func TestVllmMarshalsServerArgs(t *testing.T) {
	body, err := json.Marshal(Vllm{ServerArgs: []string{
		"--enable-auto-tool-choice",
		"--tool-call-parser",
		"hermes",
	}})
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(string(body), `"serverArgs":["--enable-auto-tool-choice","--tool-call-parser","hermes"]`) {
		t.Fatalf("expected serverArgs in %s", body)
	}
}
