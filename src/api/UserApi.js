import axios, { Axios } from 'axios';
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

        // Lấy accessToken từ localStorage
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



const UserApi = {
    
    getBooks(page = 1, size = 20) {
    return axiosInstance.get('/books', {
        params: {
            page,
            size
        }
    })
},

    getBook(id) {
        return axiosInstance.get(`/books/${id}`);
},
    login(email, password){
        return axiosInstance.post('/auth/token', {
        email,
        password
    }) 
},
   borrow(userId, bookId){
    return axiosInstance.post("books/borrow", { userId, bookId });
},
   getRecommendedBooks(userId){
    return axiosInstance.get(`/recommend?userId=${userId}`);
},
   getPendingTransactions(userId) {
    return axiosInstance.post('transactions/search', { userId, status: 'PENDING' });
  },
   
  getReturnTransactions(userId) {
    return axiosInstance.post('transactions/search', { userId, status: 'RETURNED' });
  },

 getFavoriteBooks(userId) {
  return axiosInstance.get(`/books/wishlist?userId=${userId}`);
},

  addToFavorite(userId, bookId) {
    return axiosInstance.post('/books/wishlist', { userId, bookId });
  },

  removeFromFavorite(userId, bookId) {
    return axiosInstance.delete('/books/wishlist', {
      data: { userId, bookId }
    });
  }

};




    



export default UserApi;
