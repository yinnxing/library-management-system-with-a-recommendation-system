import React from 'react';
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


const Chart = ({ data, chartTitle, label, chartColor }) => {
  const chartData = {
    labels: data.map(item => item.date),
    datasets: [
      {
        label: label || 'Data',
        data: data.map(item => item.count),
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
      {data.length === 0 ? (
        <p className={styles.noData}>No data available for the chart.</p>
      ) : (
        <Line data={chartData} />
      )}
    </div>
  );
};

export default Chart;