import React, { useState, useEffect } from 'react';
import BookList from '../../components/BookList/BookList';
import styles from './HomePage.module.css';
import { useUser } from '../../contexts/UserContext'; 
import UserApi from '../../api/UserApi';
import all_books from '../../assets/books';

const HomePage = () => {
  const [books, setBooks] = useState([]);
  const [searchResults, setSearchResults] = useState([]);
  const [recommendedBooks, setRecommendedBooks] = useState([]);
  const [popularBooks, setPopularBooks] = useState([]);
  const { user } = useUser();


  useEffect(() => {
    fetchBooks();
    // fetchRecommendBooks();
  }, []);

  const fetchBooks = async () => {
    try {
      const response = await UserApi.getBooks();
      if (response.data.code === 0) {
        const fetchBooks = response.data.result.content;
        // setBooks(fetchedBooks);

        // const popular = fetchedBooks.slice(2, 5); // Lấy 2 sách đầu tiên
        // const recommended = fetchedBooks.slice(2, 7); // Lấy 2 sách tiếp theo
        setPopularBooks(fetchBooks);
        // setRecommendedBooks(recommended);
      } else {
        console.error("Lỗi: API trả về mã lỗi không phải 0.");
      }
    } catch (error) {
      console.error("Lỗi khi lấy danh sách sách:", error);
    }
  };
  const fetchRecommendBooks = async () => {
    try {
      console.log(user.userId);
      const response = await UserApi.getRecommendedBooks("14d41a95-6161-4871-bfdf-6e29133283ea");
      const titleBooks = response.data.recommended_books;
      const recommendedBooks = all_books.filter((b) => titleBooks.includes(b.title));

      setRecommendedBooks(recommendedBooks);
      
    } catch (error) {
      console.error("Lỗi khi lấy danh sách sách:", error);
    }
  };
  

  return (
    <div className={styles.homePage}>
      <section className={styles.carouselSection}>
        <h2>Sách Phổ Biến</h2>
        <BookList books={popularBooks} />
      </section>

      {/* <section className={styles.recommendedSection}>
        <h2>Gợi Ý Dành Cho Bạn</h2>
        <BookList books={recommendedBooks} />
      </section> */}

      {searchResults.length > 0 && (
        <section className={styles.searchResultsSection}>
          <h2>Kết Quả Tìm Kiếm</h2>
          <BookList books={searchResults} />
        </section>
      )}
    </div>
  );
};

export default HomePage;
