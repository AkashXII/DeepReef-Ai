import React, { useState } from "react";
import axios from "axios";
import { motion } from "framer-motion";
import "../global.css";

export default function CoralAnalyzer() {
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [form, setForm] = useState({
    Temperature_Mean: "",
    Windspeed: "",
    TSA: "",
    Ocean_Name: "",
    Exposure: "",
  });
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);

  const handleChange = (e) => setForm({ ...form, [e.target.name]: e.target.value });
  const handleFileChange = (e) => {
    const selectedFile = e.target.files[0];
    setFile(selectedFile);
    setPreview(URL.createObjectURL(selectedFile));
  };

  const handleSubmit = async () => {
    if (!file) return alert("Please upload a coral image.");
    for (const v of Object.values(form)) if (!v) return alert("Please fill all fields.");

    setLoading(true);
    const formData = new FormData();
    formData.append("file", file);
    Object.entries(form).forEach(([key, val]) => formData.append(key, val));

    try {
      const res = await axios.post("http://localhost:8000/analyze_coral", formData);
      setResult(res.data);
    } catch (err) {
      console.error(err);
      alert("Error analyzing coral. Check backend connection.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <motion.div
      className="playfair-display-small-400 max-w-3xl mx-auto mt-12 bg-gradient-to-br from-gray-900 to-blue-950 rounded-3xl shadow-2xl p-8 border border-blue-800 text-white"
      initial={{ opacity: 0, y: 40 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.8 }}
    >
      <h2 className="text-3xl font-bold text-center mb-8 text-blue-400 tracking-wide">
        Coral Reef Analysis
      </h2>

      {/* Image upload + preview */}
      <motion.div
        className="mb-6"
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ delay: 0.3 }}
      >
        <label className="block mb-3 text-blue-300 font-semibold">
          Upload Coral Image
        </label>
        <input
          type="file"
          accept="image/*"
          onChange={handleFileChange}
          className="block w-full text-gray-300 bg-gray-800 border border-gray-700 rounded-lg p-2 cursor-pointer hover:border-blue-400 transition-all"
        />

        {preview && (
          <motion.div
            className="flex justify-center mt-5"
            initial={{ scale: 0.8, opacity: 0 }}
            animate={{ scale: 1, opacity: 1 }}
            transition={{ duration: 0.5 }}
          >
            <img
              src={preview}
              alt="Coral Preview"
              className="rounded-2xl border border-blue-700 w-64 h-64 object-cover shadow-xl"
            />
          </motion.div>
        )}
      </motion.div>

      {/* Input Fields */}
      <div className="grid md:grid-cols-2 gap-6 mb-8">
        {[
          { name: "Temperature_Mean", placeholder: "Temperature Mean (°C)" },
          { name: "Windspeed", placeholder: "Windspeed (m/s)" },
          { name: "TSA", placeholder: "TSA (Temperature Stress Anomaly)" },
        ].map((input) => (
          <motion.input
            key={input.name}
            type="number"
            step="any"
            name={input.name}
            placeholder={input.placeholder}
            value={form[input.name]}
            onChange={handleChange}
            className="bg-gray-800 border border-gray-700 rounded-lg p-3 w-full text-white placeholder-gray-400 focus:border-blue-500 focus:ring-2 focus:ring-blue-500 transition-all"
            whileFocus={{ scale: 1.03 }}
          />
        ))}

        {/* Animated Dropdowns */}
        <motion.select
          name="Ocean_Name"
          value={form.Ocean_Name}
          onChange={handleChange}
          className="bg-gray-800 border border-gray-700 rounded-lg p-3 w-full text-white focus:border-blue-500 focus:ring-2 focus:ring-blue-500 transition-all cursor-pointer"
          whileFocus={{ scale: 1.03 }}
        >
          <option value="">Select Ocean</option>
          <option value="Atlantic">Atlantic</option>
          <option value="Pacific">Pacific</option>
          <option value="Indian">Indian</option>
        </motion.select>

        <motion.select
          name="Exposure"
          value={form.Exposure}
          onChange={handleChange}
          className="bg-gray-800 border border-gray-700 rounded-lg p-3 w-full text-white focus:border-blue-500 focus:ring-2 focus:ring-blue-500 transition-all cursor-pointer"
          whileFocus={{ scale: 1.03 }}
        >
          <option value="">Select Exposure</option>
          <option value="Exposed">Exposed</option>
          <option value="Sheltered">Sheltered</option>
        </motion.select>
      </div>

      {/* Submit Button */}
      <motion.button
        onClick={handleSubmit}
        disabled={loading}
        whileHover={{ scale: loading ? 1 : 1.05 }}
        whileTap={{ scale: 0.97 }}
        className={`w-full py-3 rounded-lg font-semibold tracking-wide text-lg transition-all ${
          loading
            ? "bg-gray-500 cursor-not-allowed"
            : "bg-blue-600 hover:bg-blue-700 text-white shadow-lg"
        }`}
      >
        {loading ? "Analyzing..." : "Analyze Coral"}
      </motion.button>

      {/* Results */}
      {result && (
        <motion.div
          className="mt-8 bg-gray-800 border border-blue-800 p-6 rounded-2xl shadow-lg"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ duration: 0.7 }}
        >
          <h3 className="text-2xl font-bold mb-4 text-blue-400">🧾 Results</h3>
          <p><strong>Image Prediction:</strong> {result.image_prediction}</p>
          <p><strong>Confidence:</strong> {result.confidence}%</p>
          <p><strong>Bleaching Severity:</strong> {result.bleaching_severity}</p>

          {result.gemini_summary && (
            <div className="mt-5 p-4 bg-gray-900 rounded-lg border border-blue-700">
              <h4 className="font-semibold mb-2 text-blue-300">
                Gemini 2.5 Summary:
              </h4>
              <p className="text-gray-200 whitespace-pre-line leading-relaxed">
                {result.gemini_summary}
              </p>
            </div>
          )}
        </motion.div>
      )}
    </motion.div>
  );
}
