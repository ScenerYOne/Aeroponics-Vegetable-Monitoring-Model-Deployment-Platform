import { useState } from "react";
import { Zap, Image as ImageIcon, Brain, History } from "lucide-react";
import "./App.css";

function App() {
  const [modelFile, setModelFile] = useState(null);
  const [activeModelId, setActiveModelId] = useState(null);
  const [currentModelId, setCurrentModelId] = useState(null);
  const [testFile, setTestFile] = useState(null);
  const [testFilePreview, setTestFilePreview] = useState(null);
  const [resultImage, setResultImage] = useState(null);
  const [detections, setDetections] = useState([]);
  const [isUploadingModel, setIsUploadingModel] = useState(false);
  const [isPredicting, setIsPredicting] = useState(false);
  const [classNames, setClassNames] = useState([]);
  const [modelFormat, setModelFormat] = useState(null);

  const [modelHistory, setModelHistory] = useState([]);

  const API_BASE = import.meta.env.VITE_API_BASE || "/api";

  const handleModelUpload = async () => {
    if (!modelFile) return;
    setIsUploadingModel(true);

    const formData = new FormData();
    formData.append("file", modelFile);

    try {
      const res = await fetch(`${API_BASE}/upload-model`, {
        method: "POST",
        body: formData,
      });

      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || "Upload failed");

      setCurrentModelId(data.model_id);
      setClassNames(data.class_names);
      setModelFormat(data.model_format);
      setActiveModelId(modelFile.name);

      setModelHistory((prev) => [
        {
          model_id: data.model_id,
          name: modelFile.name,
          format: data.model_format,
          class_names: data.class_names,
        },
        ...prev,
      ]);


      setResultImage(null);
      setDetections([]);

      alert(`โหลดโมเดลสำเร็จ (${data.model_format.toUpperCase()})`);
    } catch (err) {
      alert("อัปโหลดโมเดลไม่สำเร็จ: " + err.message);
    }

    setIsUploadingModel(false);
  };

  const switchModel = async (model) => {
    await fetch(`${API_BASE}/use-model`, {
      method: "POST",
      body: new URLSearchParams({ model_id: model.model_id })
    });
  
    setCurrentModelId(model.model_id);
    setClassNames(model.class_names);
    setModelFormat(model.format);
    setActiveModelId(model.name);
  
    setResultImage(null);
    setDetections([]);
  };
  


  const handlePredict = async () => {
    if (!testFile || !currentModelId) return;
    setIsPredicting(true);

    const formData = new FormData();
    formData.append("file", testFile);
    formData.append("model_id", currentModelId); // ⭐ สำคัญมาก

    try {
      const res = await fetch(`${API_BASE}/predict`, {
        method: "POST",
        body: formData,
      });

      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || "Inference failed");

      setResultImage(`data:image/jpeg;base64,${data.image}`);
      setDetections(data.detections || []);
    } catch (err) {
      alert("ตรวจจับไม่สำเร็จ: " + err.message);
      setResultImage(testFilePreview);
    }

    setIsPredicting(false);
  };

  const handleTestFileChange = (e) => {
    const file = e.target.files[0];
    if (!file) return;

    setTestFile(file);

    const reader = new FileReader();
    reader.onloadend = () => {
      setTestFilePreview(reader.result);
      setResultImage(null);
      setDetections([]);
    };
    reader.readAsDataURL(file);
  };

  return (
    <div className="app-container">
      <div className="header">
        <div className="inner">
          <h1>Model Tester</h1>
        </div>
      </div>

      <div className="main-content">
        {/*  LEFT  */}
        <div>
          <h2 className="font-light mb-8 text-gray-300">Upload & Run</h2>

          {/* Upload Model */}
          <div className="mb-8">
            <label>
              <input
                type="file"
                accept=".pt,.onnx,.pth"
                onChange={(e) => {
                  if (e.target.files[0]) {
                    setModelFile(e.target.files[0]);
                    setActiveModelId(e.target.files[0].name);
                  }
                }}
                className="hidden"
              />
              <div className="upload-box">
                <Brain className="w-10 h-10 text-blue-500" />
                <span className="upload-text">
                  {activeModelId || "Upload Model (.pt / .onnx)"}
                </span>
              </div>
            </label>

            {modelFile && !currentModelId && (
              <button
                onClick={handleModelUpload}
                disabled={isUploadingModel}
                className="action-btn w-full mt-4"
              >
                {isUploadingModel ? "Uploading Model" : "Upload Model to Server"}
              </button>
            )}

            {currentModelId && (
              <>
                <div className="text-green-400 text-sm mt-2 text-center">
                  Model Ready to Use
                </div>

                {modelFormat && (
                  <div className="flex justify-center mt-2">
                    <span
                      className={`px-3 py-1 text-xs font-semibold rounded-full ${
                        modelFormat === "onnx"
                          ? "bg-purple-600"
                          : "bg-blue-600"
                      }`}
                    >
                      {modelFormat.toUpperCase()}
                    </span>
                  </div>
                )}
              </>
            )}
          </div>

          {/* Upload Image */}
          <div className="mb-8">
            <label>
              <input
                type="file"
                accept="image/*"
                onChange={handleTestFileChange}
                className="hidden"
              />
              <div className="upload-box">
                <ImageIcon className="w-10 h-10 text-green-500" />
                <span className="upload-text">
                  {testFile ? testFile.name : "Upload Test Image"}
                </span>
              </div>
            </label>
          </div>

          <button
            onClick={handlePredict}
            disabled={isPredicting || !testFile || !currentModelId}
            className="action-btn w-full"
          >
            <Zap className="w-10 h-10" />
            {isPredicting ? "Analyzing" : "Run Inference"}
          </button>

          {/* Model History */}
          {modelHistory.length > 1 && (
            <div className="stats-card mt-6">
              <div className="stats-title flex items-center gap-2">
                <History size={16} /> Model History
              </div>

              {modelHistory.map((m) => (
                <div
                  key={m.model_id}
                  className={`stat-item cursor-pointer ${
                    currentModelId === m.model_id ? "bg-gray-800" : ""
                  }`}
                  onClick={() => switchModel(m)}
                >
                  <span className="text-gray-200 text-sm truncate">
                    {m.name}
                  </span>
                  <span className="text-cyan-400 text-xs">
                    {m.format.toUpperCase()}
                  </span>
                </div>
              ))}
            </div>
          )}
        </div>

          {/* ================= RIGHT ================= */}
          <div className="result-zone">
            <h3>Inference Results</h3>
            {(testFilePreview || resultImage) ? (
              <>
                <div className="result-images">
                  {testFilePreview && (
                    <div className="result-card">
                      <div className="result-label">Input</div>
                      <img src={testFilePreview} alt="Input" />
                    </div>
                  )}
                  {resultImage && (
                    <div className="result-card">
                      <div className="result-label">Output</div>
                      <img src={resultImage} alt="Output" />
                    </div>
                  )}
                </div>

                {/* 🔹 Detection Summary (ของเดิมคุณ) */}
                {detections.length > 0 && (
                  <div className="stats-card">
                    <div className="stats-title">Detection Summary</div>

                    {(() => {
                      const total = detections.length;
                      const summary = detections.reduce((acc, d) => {
                        const name = classNames[d.cls] || `Class ${d.cls}`;
                        acc[name] = (acc[name] || 0) + 1;
                        return acc;
                      }, {});

                      return Object.entries(summary)
                        .sort(([, a], [, b]) => b - a)
                        .map(([name, count]) => {
                          const percentage = (
                            (count / total) *
                            100
                          ).toFixed(2);
                          return (
                            <div key={name} className="stat-item">
                              <span className="class-name text-gray-200">
                                {name}
                              </span>
                              <span className="text-cyan-400 font-medium">
                                {count} objects · {percentage}%
                              </span>
                            </div>
                          );
                        });
                    })()}

                    <div className="mt-4">
                      <div className="stat-item-no-border">
                        <span className="text-gray-400 text-sm">
                          Total Detections
                        </span>
                        <span className="text-xl font-bold text-white">
                          {detections.length}
                        </span>
                      </div>

                      <div className="stat-item-no-border">
                        <span className="text-gray-400 text-sm">
                          Model Avg Confidence
                        </span>
                        <span className="text-xl font-bold text-cyan-400">
                          {detections.length > 0
                            ? (
                                (detections.reduce(
                                  (sum, d) => sum + (d.conf || 0),
                                  0
                                ) /
                                  detections.length) *
                                100
                              ).toFixed(1)
                            : 0}
                          %
                        </span>
                      </div>
                    </div>
                  </div>
                )}
              </>
            ) : (
              <div className="text-center text-gray-500 mt-20">
                <Brain className="w-24 h-24 mx-auto mb-4 opacity-20" />
                <p>อัปโหลดโมเดลและภาพเพื่อเริ่มใช้งาน</p>
              </div>
            )}
          </div>
        </div>
      </div>
  );
}

export default App;
