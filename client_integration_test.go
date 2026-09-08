//go:build integration

package huggingface

import (
	"os"
	"testing"
	"time"
)

func integrationClient(t *testing.T) *Client {
	t.Helper()
	namespace := os.Getenv("HUGGINGFACE_NAMESPACE")
	token := os.Getenv("HUGGINGFACE_TOKEN")
	if namespace == "" || token == "" {
		t.Skip("HUGGINGFACE_NAMESPACE and HUGGINGFACE_TOKEN are required")
	}

	client, err := NewClient(nil, &namespace, &token)
	if err != nil {
		t.Fatal(err)
	}
	return client
}

func TestIntegrationCreateVllmQwenGPU(t *testing.T) {
	client := integrationClient(t)
	name := "test-vllm-qwen-" + time.Now().UTC().Format("20060102150405")
	endpoint, err := client.CreateEndpoint(CreateEndpointRequest{
		Compute: Compute{
			Accelerator:  "gpu",
			InstanceSize: "x1",
			InstanceType: "nvidia-t4",
			Scaling: Scaling{
				MinReplica: 0,
				MaxReplica: 1,
			},
		},
		Model: Model{
			Framework:  "pytorch",
			Repository: "Qwen/Qwen2.5-0.5B-Instruct",
			Image: Image{
				Vllm: &Vllm{
					URL: "vllm/vllm-openai:latest",
					ServerArgs: []string{
						"--enable-auto-tool-choice",
						"--tool-call-parser",
						"hermes",
					},
				},
			},
		},
		Name: name,
		Provider: Provider{
			Region: "us-east-1",
			Vendor: "aws",
		},
		Type: "protected",
	})
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		if err := client.DeleteEndpoint(endpoint.Name); err != nil {
			t.Errorf("delete endpoint %q: %v", endpoint.Name, err)
		}
	})
}
