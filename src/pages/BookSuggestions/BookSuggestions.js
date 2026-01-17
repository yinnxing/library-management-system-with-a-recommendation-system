import React, { useState, useEffect } from 'react';
import BookFeedback from '../../components/BookFeedback/BookFeedback';
import '../../styles/design-system.css';
import styles from './BookSuggestions.module.css';
import { useUser } from '../../contexts/UserContext';
import UserApi from '../../api/UserApi';

const BookSuggestions = () => {
  const [recommendedBooks, setRecommendedBooks] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const { user } = useUser();

  const defaultRecommendedBooks = [
    {
      "bookId": 84,
      "title": "Mystic River",
      "author": "Brian Helgeland",
      "publisher": "Default Publisher",
      "publicationYear": 2002,
      "isbn": "ISBN586400",
      "genre": "Motion picture plays",
      "descriptions": "Unmarked typescript, dated FINAL DRAFT Rev. 09/16/02 (Pink) Clint Eastwood directed, produced, and scored this mystery drama starring Sean Penn, Tim Robbins, and Kevin Bacon. It as released by Warner Brothers Oct. 3, 2003.",
      "coverImageUrl": "https://m.media-amazon.com/images/M/MV5BMTIzNDUyMjA4MV5BMl5BanBnXkFtZTYwNDc4ODM3._V1_.jpg",
      "quantity": 20,
      "availableQuantity": 2,
      "createdAt": "2024-12-30T15:27:37",
      "previewLink": "https://books.google.com.vn/books?id=WpjzzwEACAAJ&lpg=PP1&dq=intitle:Mystic+River&hl=vi&pg=PP1&output=embed"
    },
    {
      "bookId": 85,
      "title": "Sphere",
      "author": "Michael Crichton",
      "publisher": "Default Publisher",
      "publicationYear": 2011,
      "isbn": "ISBN624731",
      "genre": "Life on other planets",
      "descriptions": "No description available.",
      "coverImageUrl": "https://images-na.ssl-images-amazon.com/images/S/compressed.photo.goodreads.com/books/1660273071i/455373.jpg",
      "quantity": 8,
      "availableQuantity": 1,
      "createdAt": "2024-12-30T15:27:37",
      "previewLink": "https://books.google.com.vn/books?id=Ad6CzQEACAAJ&lpg=PP1&dq=intitle:Sphere&hl=vi&pg=PP1&output=embed"
    },
    {
      "bookId": 86,
      "title": "The Pelican Brief",
      "author": "Unknown Author",
      "publisher": "Default Publisher",
      "publicationYear": 2012,
      "isbn": "ISBN831323",
      "genre": "History",
      "descriptions": "No description available.",
      "coverImageUrl": "https://m.media-amazon.com/images/M/MV5BZjA2NmE4MjEtOTkxYy00YjhkLWI2YjgtODFmMGY0Zjc3YTdhXkEyXkFqcGc@._V1_.jpg",
      "quantity": 2,
      "availableQuantity": 16,
      "createdAt": "2024-12-30T15:27:37",
      "previewLink": "https://books.google.com.vn/books?id=-on-zwEACAAJ&lpg=PP1&dq=intitle:The+Pelican+Brief&hl=vi&pg=PP1&output=embed"
    },
  ];

  useEffect(() => {
    if (user?.userId) {
      fetchRecommendedBooks();
    } else {
      setLoading(false);
    }
  }, [user]);

  const fetchRecommendedBooks = async () => {
    try {
      setLoading(true);
      setError(null);

      if (!user || !user.userId) {
        console.error("User ID không hợp lệ");
        setRecommendedBooks(defaultRecommendedBooks);
        return;
      }

      const response = await UserApi.getRecommendedBooks(user.userId);
      
      // Check if response has the expected structure
      if (response.data.code !== 0) {
        console.error("API trả về mã lỗi:", response.data.code, response.data.message);
        setError(response.data.message || "Không thể tải danh sách gợi ý.");
        setRecommendedBooks(defaultRecommendedBooks);
        return;
      }

      const result = response.data.result;
      const recommendedBooksData = result.recommendations;

      if (!recommendedBooksData || !Array.isArray(recommendedBooksData)) {
        console.error("Dữ liệu recommendedBooks không đúng định dạng");
        setRecommendedBooks(defaultRecommendedBooks);
        return;
      }

      // The API now returns complete book data, so we can use it directly
      const recommendedBooks = recommendedBooksData.map((book) => ({
        bookId: book.bookId,
        title: book.title,
        author: book.author,
        publisher: book.publisher,
        publicationYear: book.publicationYear,
        isbn: book.isbn,
        genre: book.genre || "Recommended",
        descriptions: book.descriptions,
        coverImageUrl: book.coverImageUrl,
        quantity: book.quantity,
        availableQuantity: book.availableQuantity,
        createdAt: book.createdAt,
        previewLink: book.previewLink,
        isAvailable: book.isAvailable
      }));

      setRecommendedBooks(recommendedBooks);
      
      // Log the input book for reference
      console.log("Sách được sử dụng để gợi ý:", result.inputBook);
      
    } catch (error) {
      console.error("Lỗi khi lấy danh sách sách đề xuất:", error);
      
      // Handle error response with code and message format
      if (error.response && error.response.data) {
        const errorData = error.response.data;
        if (errorData.code && errorData.message) {
          setError(errorData.message);
        } else {
          setError("Không thể tải danh sách gợi ý. Hiển thị sách mặc định.");
        }
      } else {
        setError("Lỗi kết nối đến máy chủ. Hiển thị sách mặc định.");
      }
      
      setRecommendedBooks(defaultRecommendedBooks);
    } finally {
      setLoading(false);
    }
  };



  if (!user) {
    return (
      <div className={styles.container}>
        <div className={styles.loginPrompt}>
          <div className={styles.loginIcon}>🔐</div>
          <h2>Đăng nhập để xem gợi ý sách</h2>
          <p>Vui lòng đăng nhập để nhận được những gợi ý sách phù hợp với sở thích của bạn.</p>
        </div>
      </div>
    );
  }

  return (
    <div className={styles.container}>
      {/* Header Section */}
      <div className={styles.headerSection}>
        <div className={styles.titleContainer}>
          <h1 className={styles.pageTitle}>
            <span className={styles.titleIcon}>🎯</span>
            Gợi Ý Sách Dành Cho Bạn
          </h1>
          <p className={styles.pageSubtitle}>
            Khám phá những cuốn sách gợi ý dựa trên lịch sử đặt trước và danh sách yêu thích của bạn
          </p>
        </div>
      </div>

        


      {/* Error Message */}
      {error && (
        <div className={styles.errorMessage}>
          <span className={styles.errorIcon}>⚠️</span>
          {error}
        </div>
      )}

      {/* Loading State */}
      {loading ? (
        <div className={styles.loadingContainer}>
          <div className={styles.loadingSpinner}></div>
          <p className={styles.loadingText}>Đang tạo gợi ý sách cho bạn...</p>
        </div>
      ) : (
        <>
          {/* Instructions Section */}
          <div className={styles.instructionsSection}>
            <h3 className={styles.instructionsTitle}>
              <span className={styles.instructionsIcon}>💡</span>
              Cách hoạt động của hệ thống gợi ý
            </h3>
            <div className={styles.instructionsList}>
              <div className={styles.instructionItem}>
                <span className={styles.instructionIcon}>1️⃣</span>
                <span>Hệ thống gợi ý sách dựa trên các cuốn sách bạn đã đặt trước</span>
              </div>
              <div className={styles.instructionItem}>
                <span className={styles.instructionIcon}>2️⃣</span>
                <span>Nếu chưa có lịch sử đặt trước, sẽ gợi ý dựa trên danh sách yêu thích</span>
              </div>
              <div className={styles.instructionItem}>
                <span className={styles.instructionIcon}>3️⃣</span>
                <span>Thêm sách vào danh sách yêu thích để nhận được gợi ý tương tự</span>
              </div>
            </div>
          </div>

          {/* Books Section */}
          <div className={styles.booksSection}>
            <div className={styles.sectionHeader}>
              <h2 className={styles.sectionTitle}>
                <span className={styles.sectionIcon}>📖</span>
                Danh sách gợi ý ({recommendedBooks.length} cuốn sách)
              </h2>
              <p className={styles.sectionDescription}>
                Những cuốn sách này được gợi ý dựa trên lịch sử đặt trước và danh sách yêu thích của bạn
              </p>
            </div>
            
            <div className={styles.booksContainer}>
              <BookFeedback 
                books={recommendedBooks} 
                userId={user?.userId}
              />
            </div>
          </div>

          {/* Tips Section */}
          <div className={styles.tipsSection}>
            <h3 className={styles.tipsTitle}>
              <span className={styles.tipsIcon}>💭</span>
              Mẹo để nhận được gợi ý tốt hơn
            </h3>
            <div className={styles.tipsList}>
              <div className={styles.tipItem}>
                <span className={styles.tipIcon}>📋</span>
                <span>Đặt trước nhiều sách để hệ thống có thể gợi ý sách tương tự</span>
              </div>
              <div className={styles.tipItem}>
                <span className={styles.tipIcon}>❤️</span>
                <span>Thêm sách vào danh sách yêu thích để nhận gợi ý phù hợp</span>
              </div>

            </div>
          </div>
        </>
      )}
    </div>
  );
};

export default BookSuggestions; 