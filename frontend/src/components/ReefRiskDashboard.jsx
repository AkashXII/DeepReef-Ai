"use client";
import React, { useEffect, useState } from "react";
import axios from "axios";
import { CardBody, CardContainer, CardItem } from "../components/ui/3d-card";

export default function ReefRiskDashboard() {
  const [reefs, setReefs] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  // Random coral reef images
  const reefImages = [
    "https://www.relaxingjourneys.co.nz/wp-content/uploads/2024/10/the-great-barrier-reef.jpg",
    "https://faw-marketing.transforms.svdcdn.com/production/images/Healthy-Coral-and-Fish.jpg?w=1072&h=715&auto=compress%2Cformat&fit=crop&crop=focalpoint&fp-x=0.3776&fp-y=0.4151&dm=1547312255&s=ff62bdb7c8b2395aa01fdc68089aeb08",
    "https://images.indianexpress.com/2017/03/coral-reef-759.jpg?w=414",
    "https://files.worldwildlife.org/wwfcmsprod/images/Bottlenose_Dolphin_Belize_Reef/story_full_width/9igt6szp0c_Large_WW22277.jpg",
    "https://static-cdn.toi-media.com/www/uploads/2024/11/DSC04479.jpg",
    "https://images.unsplash.com/photo-1564866657314-2d31d89b9e2d",
  ];

  useEffect(() => {
    const fetchReefData = async () => {
      try {
        const res = await axios.get("https://w16fp9rg-8000.inc1.devtunnels.ms/predict_top5");
        setReefs(res.data.results);
      } catch (err) {
        console.error(err);
        setError("Failed to fetch reef data");
      } finally {
        setLoading(false);
      }
    };

    fetchReefData();
    const interval = setInterval(fetchReefData, 10 * 60 * 1000);
    return () => clearInterval(interval);
  }, []);

  if (loading) {
    return <div className="text-center text-blue-300 mt-10">🌊 Fetching real-time reef data...</div>;
  }

  if (error) {
    return <div className="text-center text-red-400 mt-10">❌ {error}</div>;
  }

  return (
    <div className="mt-10 p-8 min-h-screen bg-gradient-to-br  text-white playfair-display-small-400">
      <h2 className="text-3xl font-bold text-center text-blue-400 mb-0">
        Global Coral Bleaching Risk Monitor
      </h2>

      {/* Grid layout with spacing between cards */}
      <div className="grid sm:grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-96 gap-y-10 mb-10">
        {/* ↑ gap-x controls horizontal spacing; gap-y controls vertical spacing */}

        {reefs.map((reef, index) => {
          const image = reefImages[index % reefImages.length];
          return (
            <CardContainer key={index}>
              <CardBody className="relative group/card bg-gray-900 border border-blue-800/40 rounded-2xl shadow-lg hover:shadow-blue-500/30 transition-all duration-300 p-6">
                <CardItem
                  translateZ="70"
                  className="text-xl font-semibold text-blue-300"
                >
                  {reef.reef}
                </CardItem>

                <CardItem as="p" translateZ="60" className="text-gray-400 text-sm mt-2">
                  📍 Coordinates: [{reef.coordinates[0]}, {reef.coordinates[1]}]
                </CardItem>

                <CardItem translateZ="120" className="mt-4 rounded-lg overflow-hidden">
                  <img
                    src={image}
                    alt={reef.reef}
                    className="h-48 w-full object-cover rounded-xl group-hover/card:shadow-xl"
                  />
                </CardItem>

                <div className="mt-5 space-y-2">
                  <p>
                    <strong>Bleaching Risk:</strong>{" "}
                    <span
                      className={`font-semibold ${
                        reef.bleaching_risk === "Severe"
                          ? "text-red-500"
                          : reef.bleaching_risk === "Moderate"
                          ? "text-yellow-400"
                          : "text-green-400"
                      }`}
                    >
                      {reef.bleaching_risk}
                    </span>
                  </p>

                  <p className="text-sm text-gray-400">
                    Confidence: {(reef.probability * 100).toFixed(1)}%
                  </p>

                  <div className="bg-gray-800 p-3 rounded-lg border border-blue-800/40 text-sm mt-4 space-y-1">
                    <p>🌡️ Temp (max): {reef.factors.temperature_2m_max.toFixed(2)} °C</p>
                    <p>☀️ UV Index (max): {reef.factors.uv_index_max.toFixed(2)}</p>
                    <p>💨 Wind Speed (max): {reef.factors.wind_speed_10m_max.toFixed(2)} km/h</p>
                    <p>🌧️ Precipitation: {reef.factors.precipitation_sum.toFixed(2)} mm</p>
                  </div>
                </div>
              </CardBody>
            </CardContainer>
          );
        })}
      </div>
    </div>
  );
}
