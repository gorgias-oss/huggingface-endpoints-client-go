package huggingface

import (
	"fmt"
	"math/rand"
	"testing"
	"time"
)

var client *Client

func TestMain(m *testing.M) {
	host := "https://api.endpoints.huggingface.cloud/v2/endpoint"
	namespace := "issamemari"
	token := ""

	var err error
	client, err = NewClient(&host, &namespace, &token)
	if err != nil {
		panic(err)
	}

	m.Run()
}

func randomString(n int) string {
	const letters = "abcdefghijklmnopqrstuvwxyz"
	rng := rand.New(rand.NewSource(time.Now().UnixNano()))
	b := make([]byte, n)
	for i := range b {
		b[i] = letters[rng.Intn(len(letters))]
	}
	return string(b)
}

func newCreateEndpointRequest() CreateEndpointRequest {
	name := fmt.Sprintf("test-endpoint-%s", randomString(4))
	scaleToZeroTimeout := 15
	revision := "main"
	task := "sentence-embeddings"
	pendingRequests := 1.5
	return CreateEndpointRequest{
		AccountId: nil,
		Compute: Compute{
			Accelerator:  "cpu",
			InstanceSize: "x4",
			InstanceType: "intel-icl",
			Scaling: Scaling{
				MinReplica: 0,
				MaxReplica: 1,
				Measure: &Measure{
					PendingRequests: &pendingRequests,
					HardwareUsage:   nil,
				},
				ScaleToZeroTimeout: &scaleToZeroTimeout,
			},
		},
		Model: Model{
			Framework: "pytorch",
			Image: Image{
				Huggingface: &Huggingface{},
			},
			Repository: "sentence-transformers/all-MiniLM-L6-v2",
			Revision:   &revision,
			Task:       &task,
			Env:        map[string]string{},
		},
		Name: name,
		Provider: Provider{
			Region: "us-east-1",
			Vendor: "aws",
		},
		Type: "protected",
	}
}

func newCreateEndpointRequestWithCustomImage() CreateEndpointRequest {
	endpoint := newCreateEndpointRequest()
	endpoint.Model.Image.Custom = &Custom{
		Credentials: &Credentials{
			Password: "password",
			Username: "username",
		},
		HealthRoute: nil,
		Port:        nil,
		URL:         "https://example.com",
	}
	endpoint.Model.Env = map[string]string{
		"key": "value",
	}
	endpoint.Model.Image.Huggingface = nil
	return endpoint
}

func TestCustomImage(t *testing.T) {
	endpoint := newCreateEndpointRequestWithCustomImage()

	_, err := client.CreateEndpoint(endpoint)
	if err != nil {
		panic(err)
	}

	err = client.DeleteEndpoint(endpoint.Name)
	if err != nil {
		panic(err)
	}
}

func TestNilCredentials(t *testing.T) {
	endpoint := newCreateEndpointRequestWithCustomImage()
	endpoint.Model.Image.Custom.Credentials = nil

	_, err := client.CreateEndpoint(endpoint)
	if err != nil {
		panic(err)
	}

	err = client.DeleteEndpoint(endpoint.Name)
	if err != nil {
		panic(err)
	}
}

func TestEmptyEnv(t *testing.T) {
	endpoint := newCreateEndpointRequest()
	endpoint.Model.Env = map[string]string{}

	_, err := client.CreateEndpoint(endpoint)
	if err != nil {
		panic(err)
	}

	err = client.DeleteEndpoint(endpoint.Name)
	if err != nil {
		panic(err)
	}
}

func TestListEndpoints(t *testing.T) {
	_, err := client.ListEndpoints()
	if err != nil {
		panic(err)
	}
}

func TestCreateAndDeleteEndpoint(t *testing.T) {
	endpoint := newCreateEndpointRequest()

	_, err := client.CreateEndpoint(endpoint)
	if err != nil {
		panic(err)
	}

	err = client.DeleteEndpoint(endpoint.Name)
	if err != nil {
		panic(err)
	}
}

func TestGetEndpoint(t *testing.T) {
	endpoint := newCreateEndpointRequest()

	_, err := client.CreateEndpoint(endpoint)
	if err != nil {
		panic(err)
	}

	_, err = client.GetEndpoint(endpoint.Name)
	if err != nil {
		panic(err)
	}

	err = client.DeleteEndpoint(endpoint.Name)
	if err != nil {
		panic(err)
	}
}

func TestUpdateEndpoint(t *testing.T) {
	endpoint := newCreateEndpointRequest()

	_, err := client.CreateEndpoint(endpoint)
	if err != nil {
		panic(err)
	}

	updateEndpointRequest := UpdateEndpointRequest{
		Compute: &Compute{
			Accelerator:  "cpu",
			InstanceSize: "x8",
			InstanceType: "intel-icl",
			Scaling: Scaling{
				MinReplica: 0,
				MaxReplica: 1,
			},
		},
		Model: &endpoint.Model,
		Type:  nil,
	}

	_, err = client.UpdateEndpoint(endpoint.Name, updateEndpointRequest)
	if err != nil {
		panic(err)
	}

	err = client.DeleteEndpoint(endpoint.Name)
	if err != nil {
		panic(err)
	}
}

func TestOptionalFields(t *testing.T) {
	endpoint := newCreateEndpointRequest()
	endpoint.Model.Revision = nil
	endpoint.Compute.Scaling.ScaleToZeroTimeout = nil

	_, err := client.CreateEndpoint(endpoint)
	if err != nil {
		panic(err)
	}

	err = client.DeleteEndpoint(endpoint.Name)
	if err != nil {
		panic(err)
	}
}

func TestTeiImage(t *testing.T) {
	endpoint := newCreateEndpointRequest()
	endpoint.Model.Image.Huggingface = nil
	endpoint.Model.Image.Tei = &Tei{
		URL:                   "ghcr.io/huggingface/text-embeddings-inference:1.2",
		MaxBatchTokens:        &[]int{8192}[0],
		MaxConcurrentRequests: &[]int{512}[0],
		Pooling:               &[]string{"mean"}[0],
	}

	_, err := client.CreateEndpoint(endpoint)
	if err != nil {
		panic(err)
	}

	err = client.DeleteEndpoint(endpoint.Name)
	if err != nil {
		panic(err)
	}
}

func TestTgiImage(t *testing.T) {
	endpoint := newCreateEndpointRequest()
	endpoint.Model.Image.Huggingface = nil
	endpoint.Model.Image.Tgi = &Tgi{
		URL:                   "ghcr.io/huggingface/text-generation-inference:1.4",
		MaxBatchPrefillTokens: &[]int{4096}[0],
		MaxBatchTotalTokens:   &[]int{8192}[0],
		MaxInputLength:        &[]int{4096}[0],
		MaxTotalTokens:        &[]int{8192}[0],
		Quantize:              &[]string{"bitsandbytes"}[0],
	}

	_, err := client.CreateEndpoint(endpoint)
	if err != nil {
		panic(err)
	}

	err = client.DeleteEndpoint(endpoint.Name)
	if err != nil {
		panic(err)
	}
}

func TestTgiNeuronImage(t *testing.T) {
	endpoint := newCreateEndpointRequest()
	endpoint.Model.Image.Huggingface = nil
	endpoint.Model.Image.TgiNeuron = &TgiNeuron{
		URL:                   "ghcr.io/huggingface/neuronx-tgi:0.0.15",
		MaxBatchPrefillTokens: &[]int{4096}[0],
		MaxBatchTotalTokens:   &[]int{8192}[0],
		MaxInputLength:        &[]int{4096}[0],
		MaxTotalTokens:        &[]int{8192}[0],
		HfAutoCastType:        &[]string{"bf16"}[0],
		HfNumCores:            &[]int{2}[0],
	}

	_, err := client.CreateEndpoint(endpoint)
	if err != nil {
		panic(err)
	}

	err = client.DeleteEndpoint(endpoint.Name)
	if err != nil {
		panic(err)
	}
}

func TestLlamacppImage(t *testing.T) {
	endpoint := newCreateEndpointRequest()
	endpoint.Model.Image.Huggingface = nil
	endpoint.Model.Image.Llamacpp = &Llamacpp{
		URL:         "ghcr.io/ggerganov/llama.cpp:server",
		ModelPath:   "/app/model.gguf",
		CtxSize:     &[]int{4096}[0],
		Embeddings:  &[]bool{false}[0],
		NParallel:   &[]int{1}[0],
		ThreadsHttp: &[]int{4}[0],
	}

	_, err := client.CreateEndpoint(endpoint)
	if err != nil {
		panic(err)
	}

	err = client.DeleteEndpoint(endpoint.Name)
	if err != nil {
		panic(err)
	}
}

func TestVllmImage(t *testing.T) {
	endpoint := newCreateEndpointRequest()
	endpoint.Model.Image.Huggingface = nil
	endpoint.Model.Image.Vllm = &Vllm{
		URL:                 "vllm/vllm-openai:latest",
		KvCacheDtype:        &[]string{"auto"}[0],
		MaxNumBatchedTokens: &[]int{8192}[0],
		MaxNumSeqs:          &[]int{256}[0],
		TensorParallelSize:  &[]int{1}[0],
	}

	_, err := client.CreateEndpoint(endpoint)
	if err != nil {
		panic(err)
	}

	err = client.DeleteEndpoint(endpoint.Name)
	if err != nil {
		panic(err)
	}
}
