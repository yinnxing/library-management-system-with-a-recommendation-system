import React, { useState, useEffect } from 'react';
import { Line } from 'react-chartjs-2';
import {
  Chart as ChartJS,
  Title,
  Tooltip,
  Legend,
  LineElement,
  CategoryScale,
  LinearScale,
  PointElement,
} from 'chart.js';
import styles from './Chart.module.css';

ChartJS.register(Title, Tooltip, Legend, LineElement, CategoryScale, LinearScale, PointElement);

// Mock Data
const mockChartData = [
  { date: '2025-01-01', count: 12 },
  { date: '2025-01-02', count: 19 },
  { date: '2025-01-03', count: 7 },
  { date: '2025-01-04', count: 15 },
  { date: '2025-01-05', count: 10 },
  { date: '2025-01-06', count: 22 },
  { date: '2025-01-07', count: 5 },
];

const Chart = ({ useMockData = true, fetchDataUrl, chartTitle, label, chartColor }) => {
  const [chartData, setChartData] = useState([]);
  const [error, setError] = useState(null);

  useEffect(() => {
    if (useMockData) {
      // Sử dụng dữ liệu giả
      setChartData(mockChartData);
    } else if (fetchDataUrl) {
      const fetchChartData = async () => {
        try {
          const response = await fetch(fetchDataUrl);
          if (!response.ok) {
            throw new Error(`HTTP error! Status: ${response.status}`);
          }
          const data = await response.json();
          setChartData(data.transactions || data);
        } catch (err) {
          setError(err.message);
          console.error('Error fetching chart data:', err);
        }
      };

      fetchChartData();
    }
  }, [fetchDataUrl, useMockData]);

  const data = {
    labels: chartData.map((item) => item.date),
    datasets: [
      {
        label: label || 'Data',
        data: chartData.map((item) => item.count),
        borderColor: chartColor || 'rgba(75,192,192,1)',
        backgroundColor: chartColor ? `${chartColor}0.2` : 'rgba(75,192,192,0.2)',
        fill: true,
        tension: 0.3,
        pointRadius: 3,
      },
    ],
  };

  return (
    <div className={styles.chartContainer}>
      <h3 className={styles.chartTitle}>{chartTitle}</h3>
      {error ? (
        <p className={styles.error}>Failed to load data: {error}</p>
      ) : chartData.length === 0 ? (
        <p className={styles.noData}>No data available for the chart.</p>
      ) : (
        <Line data={data} />
      )}
    </div>
  );
};

export default Chart;
