import axios from 'axios';
import { jwtDecode } from 'jwt-decode';

const API_BASE_URL = 'http://localhost:8080/api';

const axiosInstance = axios.create({
    baseURL: API_BASE_URL,
    timeout: 10000,
    headers: {
        'Content-Type': 'application/json',
    },
    withCredentials: true,
});

export const refreshToken = async () => {
    try {
        const response = await axiosInstance.post('/auth/refresh');
        const { accessToken } = response.data.result;
        localStorage.setItem('accessToken', accessToken);
        return accessToken;
    } catch (error) {
        console.error('Error refreshing token:', error.response?.data || error.message);
        throw error;
    }
};

axiosInstance.interceptors.request.use(
    async (config) => {
        if (config.url.includes('/auth/refresh')) {
            return config;
        }

        const token = localStorage.getItem('accessToken');

        if (token) {
            try {
                const decodedToken = jwtDecode(token);
                const isTokenExpired = decodedToken.exp * 1000 < Date.now();

                if (isTokenExpired) {
                    const newAccessToken = await refreshToken();
                    config.headers['Authorization'] = `Bearer ${newAccessToken}`;
                } else {
                    config.headers['Authorization'] = `Bearer ${token}`;
                }
            } catch (error) {
                console.error('Invalid token', error);
                throw new Error('Invalid token');
            }
        }

        return config;
    },
    (error) => {
        return Promise.reject(error);
    }
);

axiosInstance.interceptors.response.use(
    (response) => response,
    (error) => {
        if (error.response && error.response.status === 401) {
            console.error('Unauthorized error', error);
        }
        return Promise.reject(error);
    }
);

const AdminApi = {
    getTransactions(transactionCriteria = {}, paginationCriteria = { page: 1, size: 10 }) {
        return axiosInstance.get('/transactions', {
            params: {
                ...transactionCriteria,
                ...paginationCriteria
            }
        });
    },

    createBook(request) {
        return axiosInstance.post('admin/book', request);
    },
    deleteBook(bookId) {
        return axiosInstance.delete(`admin/book/${bookId}`);
    },
    editBook(bookId, request) {
        return axiosInstance.put(`admin/${bookId}`, request);
    },
    getBooks(page = 1, size = 20) {
        return axiosInstance.get('/books', {
            params: {
                page,
                size
            }
        }); 
    },
    // eslint-disable-next-line no-dupe-keys
    getTransactions(transactionCriteria = {}, paginationCriteria = { page: 1, size: 10 }) {
            return axiosInstance.get('admin/transactions', {
                params: {
                    ...transactionCriteria,
                    ...paginationCriteria
                }
            });
        },
    updateTransactionToBorrowed(transactionId) {
        return axiosInstance.put(`admin/transactions/${transactionId}/update-borrowed`)
        },

    updateTransactionToReturned(transactionId) {
        return axiosInstance.put(`admin/transactions/${transactionId}/update-returned`)       
    },

     getUsers() {
        return axiosInstance.get('/users');  
    },

    
};

export default AdminApi;





